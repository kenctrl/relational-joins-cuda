#pragma once
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <vector>
#include <tuple>
#include <string>
#include <type_traits>

#include <cuda.h>
#include <cub/cub.cuh> 
#include <cub/block/block_shuffle.cuh>
#include <cuda_pipeline.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include "tuple.cuh"
#include "utils.cuh"

template<typename idx_t, typename cnt_t>
struct merge_range {
    idx_t a_begin;
    idx_t a_end;
    idx_t b_begin;
    idx_t b_end;

    __device__ merge_range(const idx_t w, const idx_t x, const idx_t y, const idx_t z) 
    : a_begin(w)
    , a_end(x) 
    , b_begin(y)
    , b_end(z) {}

    __device__ merge_range(const int cta, const int spacing, const cnt_t a_count, const cnt_t b_count, const idx_t* partitions) {
        a_begin = partitions[cta];
        a_end   = partitions[cta+1];
        int diag0 = cta*spacing;
        int diag1 = min(a_count+b_count, diag0+spacing);
        b_begin = diag0-a_begin;
        b_end   = diag1-a_end;
    }
    
    __device__ __forceinline__ cnt_t a_count() { return a_end - a_begin; }
    __device__ __forceinline__ cnt_t b_count() { return b_end - b_begin; }
    __device__ __forceinline__ cnt_t total() { return a_count() + b_count(); }

    __device__ __forceinline__ bool a_valid() { return a_end > a_begin; }
    __device__ __forceinline__ bool b_valid() { return b_end > b_begin; }


    __device__ __forceinline__ merge_range align_to_zero() { 
        return merge_range(0, a_count(), a_count(), total());
    }

    __device__ __forceinline__ merge_range partition(int mp0, int diag0, int mp1, int diag1) {
        return merge_range(a_begin + mp0, 
                           a_begin + mp1, 
                           b_begin + diag0 - mp0, 
                           b_begin + diag1 - mp1);
    }
};

template<typename key_t>
__device__ __forceinline__ int merge_path(key_t* a_keys, int a_count, key_t* b_keys, int b_count, int diag) {
    int begin = max(0, diag - b_count);
    int end = min(diag, a_count);

    while(begin < end) {
        int mid = (begin + end) / 2;
        key_t a_key = a_keys[mid];
        key_t b_key = b_keys[diag - 1 - mid];
        bool pred = (a_key <= b_key);

        if(pred) begin = mid + 1;
        else end = mid;
    }
    return begin;
}

template<typename key_t>
__global__ void merge_path_partition(key_t* a, const int a_count, 
                                     key_t* b, const int b_count,
                                     int spacing, 
                                     int* partition, const int n) {
    int p = get_cuda_tid();
    if(p >= n) return;
    partition[p] = merge_path(a, a_count, b, b_count, min(spacing*p, a_count+b_count));
}

template<int NT = 512, 
         int VT = 4, 
         typename key_t,
         typename RValIt, 
         typename SValIt, 
         typename ROutIt,
         typename SOutIt>
__global__ void merge_keys_mp_improv(key_t* l_sorted_keys, key_t* r_sorted_keys, 
                              RValIt  l_sorted_vals, SValIt r_sorted_vals,
                              const int nl, const int nr, const int* partitions,
                              key_t* keys_out, ROutIt l_out, SOutIt r_out, 
                              int* g_counter, int circular_buffer_size, bool agg_only = false) {
    auto cta = blockIdx.x;
    constexpr int NV = NT*VT;
    const int tile_size = min(NV, (nl+nr)-cta*NV);

    // obtain the merge partition belonging to this thread block
    merge_range mr_global(cta, NV, nr, nl, partitions);

    auto a = r_sorted_keys + mr_global.a_begin; // needles
    auto b = l_sorted_keys + mr_global.b_begin; // haystacks

    __shared__ key_t keys[NV+1];

    for(int i = 0; i < VT; i++) {
        auto index = i*NT+threadIdx.x;
        if(index >= mr_global.total()) break;
        if(index < mr_global.a_count()) __pipeline_memcpy_async(&keys[index], &a[index], sizeof(key_t));
        else __pipeline_memcpy_async(&keys[index], &b[index - mr_global.a_count()], sizeof(key_t));
    }
    __pipeline_commit();

    // partition the workloads within the thread block
    auto mr_cta = mr_global.align_to_zero();
    int diag0 = min(mr_cta.total(), VT*threadIdx.x);
    int diag1 = min(mr_cta.total(), diag0+VT);
    int mp0[1];
    int mp1[1];
    __pipeline_wait_prior(0);
    __syncthreads();
    mp0[0] = merge_path(keys, mr_cta.a_count(), keys+mr_cta.b_begin, mr_cta.b_count(), diag0);
    mp1[0] = mr_cta.a_count();
    
    cub::BlockShuffle<int, NT>().Down(mp0, mp1);
    
    // Find the lower bound for each "needle"
    // if there is a match, the lower bound should be the first match
    // otherwise the lower bound is the smallest element in "haystack" that is
    // greater than the "needle".
    auto mr_thread = mr_cta.partition(mp0[0], diag0, mp1[0], diag1);

    __shared__ int match[VT][NT];
    __shared__ int cnt[VT][NT];
    
    #pragma unroll
    for(int i = 0; i < VT; i++) {
        match[i][threadIdx.x] = mr_thread.b_end;
        cnt[i][threadIdx.x] = 0;
    }
    
    const bool is_valid_range = mr_thread.a_valid();
    const auto a_first = mr_thread.a_begin;
    if(mp0[0] < mr_global.a_count()) {
        while(mr_thread.a_valid() && mr_thread.b_valid()) {
            auto a_key = keys[mr_thread.a_begin], b_key = keys[mr_thread.b_begin];
            if(a_key <= b_key) {
                if(a_key == b_key) {
                    match[mr_thread.a_begin - a_first][threadIdx.x] = mr_thread.b_begin;
                }
                ++mr_thread.a_begin;
            } else if(a_key > b_key) {
                ++mr_thread.b_begin;
            }
        }
    }

    // materialize the result
    if(mp0[0] < mr_global.a_count()) {
        #pragma unroll
        for(int a_index = a_first; a_index < mr_thread.a_end; a_index++) {
            int b_index = match[a_index-a_first][threadIdx.x] - mr_cta.a_count();
            while(b_index + mr_global.b_begin < nl) {
                auto b_index_cta = b_index + mr_cta.a_count();
                key_t bk = (b_index_cta < mr_cta.total()) ? keys[b_index_cta] : b[b_index];

                if(bk == keys[a_index]) {
                    b_index++;
                    cnt[a_index-a_first][threadIdx.x]++;
                } else break;
            }
        }
    }
    
    int i = 0, j = 0;
    int shuffle_ptr = 0;
    const int wid = threadIdx.x / 32;
    const int lid = threadIdx.x % 32; 
    const int threadmask = (lid < 31)? ~((1 << (lid+1)) - 1) : 0;
    
    constexpr int SHUFFLE_SIZE = 32;
    __shared__ key_t keys_shfl[NT/32][SHUFFLE_SIZE];
    __shared__ std::remove_pointer_t<ROutIt> r_shfl[NT/32][SHUFFLE_SIZE];
    __shared__ std::remove_pointer_t<SOutIt> l_shfl[NT/32][SHUFFLE_SIZE];
    
    const bool valid_range = (mp0[0] < mr_global.a_count());
    int wr_intention;
    int ptr;


    while(true) {
        while(i < VT && j >= cnt[i][threadIdx.x]) {
            i++; 
            j = 0;
        }     
        wr_intention = (valid_range && i < VT);
        
        int mask = __ballot_sync(__activemask(), wr_intention);
        if(mask == 0) break; // no one has the write intention anymore, exit
        
        int wr_offset = shuffle_ptr +  __popc(mask & threadmask);
        shuffle_ptr = shuffle_ptr + __popc(mask);

        // the buffer will be full if the current batch is written
        while (shuffle_ptr >= SHUFFLE_SIZE) {
            if (wr_intention && (wr_offset < SHUFFLE_SIZE)) {
                int a_index = a_first + i + mr_global.a_begin;
                int b_index = match[i][threadIdx.x] - mr_cta.a_count() + mr_global.b_begin + j;
                keys_shfl[wid][wr_offset] = keys[a_first+i];
                r_shfl[wid][wr_offset] = r_sorted_vals[a_index];
                l_shfl[wid][wr_offset] = l_sorted_vals[b_index];
                wr_intention = 0;
            }

            if (lid == 0) {
                ptr = atomicAdd(g_counter, SHUFFLE_SIZE);
            }

            ptr = __shfl_sync(__activemask(), ptr, 0);

            auto w_pos = (ptr + lid) % circular_buffer_size;

            keys_out[w_pos] = keys_shfl[wid][lid];
            r_out[w_pos] = r_shfl[wid][lid];
            l_out[w_pos] = l_shfl[wid][lid];

            wr_offset -= SHUFFLE_SIZE;
            shuffle_ptr -= SHUFFLE_SIZE;
        }

        if(wr_intention && (wr_offset >= 0)) {
            int a_index = a_first + i + mr_global.a_begin;
            int b_index = match[i][threadIdx.x] - mr_cta.a_count() + mr_global.b_begin + j;
            keys_shfl[wid][wr_offset] = keys[a_first+i];
            r_shfl[wid][wr_offset] = r_sorted_vals[a_index];
            l_shfl[wid][wr_offset] = l_sorted_vals[b_index];
            wr_intention = 0;
        }

        j++;
    }

    if (lid == 0) {
        ptr = atomicAdd(g_counter, shuffle_ptr);
    }

    ptr = __shfl_sync(__activemask(), ptr, 0);

    if (lid < shuffle_ptr) {
        auto w_pos = (ptr + lid) % circular_buffer_size;
        r_out[w_pos] = r_shfl[wid][lid];
        l_out[w_pos] = l_shfl[wid][lid];
        keys_out[w_pos] = keys_shfl[wid][lid];
    }
}

template<typename KeyIt, 
         typename RValIt, 
         typename SValIt, 
         typename ROutIt,
         typename SOutIt, 
         int NT = 512, 
         int VT = 4>
void merge_path(KeyIt r_sorted_keys, KeyIt s_sorted_keys, 
                RValIt r_sorted_vals, SValIt s_sorted_vals, 
                const int nr, const int ns, 
                KeyIt keys_out, ROutIt r_out, SOutIt s_out, 
                int& n_matches, int circular_buffer_size) {
    int n = (nr + ns + NT*VT-1)/(NT*VT) + 1;
    int* partitions, *g_counter;
    allocate_mem(&partitions, false, sizeof(int)*n);
    allocate_mem(&g_counter, true);

    merge_path_partition<<<num_tb(n, 128), 128>>>(s_sorted_keys, ns,
                                                    r_sorted_keys, nr,
                                                    NT*VT,
                                                    partitions, n);

    merge_keys_mp_improv<NT, VT><<<num_tb(nr+ns, NT, VT), NT>>>(r_sorted_keys, s_sorted_keys, 
                                                            r_sorted_vals, s_sorted_vals,
                                                            nr, ns,
                                                            partitions, keys_out, 
                                                            r_out, s_out, 
                                                            g_counter, circular_buffer_size);

    int temp;
    cudaMemcpy(&temp, g_counter, sizeof(int), cudaMemcpyDefault);
    n_matches = temp;

    release_mem(partitions);
    release_mem(g_counter);
}




// Use constexpr for block/thread counts if fixed
constexpr int NUM_THREADS = 128*4;
constexpr int NUM_BLOCKS = 48*2;

// Use restrict pointers and inline where beneficial
template<typename key_t>
__forceinline__ __device__ int merge_path_device(const key_t* __restrict__ a_keys, int a_count, 
                                                 const key_t* __restrict__ b_keys, int b_count, 
                                                 int diag) {
    int begin = max(0, diag - b_count);
    int end = min(diag, a_count);

    while (begin < end) {
        int mid = (begin + end) >> 1;
        key_t a_key = a_keys[mid];
        key_t b_key = b_keys[diag - 1 - mid];
        bool pred = (a_key <= b_key);
        begin = pred ? (mid + 1) : begin;
        end   = pred ? end : mid;
    }
    return begin;
}

template<typename key_t>
__global__ void create_merge_partitions(const key_t * __restrict__ r_sorted_keys, const int nr, 
                                        const key_t * __restrict__ s_sorted_keys, const int ns, 
                                        int * __restrict__ partition_starts, const int partition_size) {
    // OPTIMIZATION: Using __restrict__, no code changes to logic
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int diag_sum = threadId * partition_size;
    if (diag_sum < nr + ns) {
        int r_low = max(0, diag_sum - ns);
        int r_max = min(nr, diag_sum);

        while (r_low < r_max) {
            int r_mid = (r_low + r_max) >> 1;
            // Precompute indices carefully
            int r_index = r_mid;
            int s_index = diag_sum - r_mid - 1;

            key_t r_key_val = (r_index < nr) ? r_sorted_keys[r_index] : (key_t)999999999; // large sentinel
            key_t s_key_val = (s_index >= 0 && s_index < ns) ? s_sorted_keys[s_index] : (key_t)-999999999; // small sentinel

            // Binary search condition
            if (r_key_val <= s_key_val) {
                r_low = r_mid + 1;
            } else {
                r_max = r_mid;
            }
        }

        partition_starts[2 * threadId]     = r_low;
        partition_starts[2 * threadId + 1] = diag_sum - r_low;
    } else {
        partition_starts[2 * threadId]     = nr;
        partition_starts[2 * threadId + 1] = ns;
    }
}

template<typename key_t>
__global__ void sequential_merge(const key_t * __restrict__ r_sorted_keys, const int nr, 
                                 const key_t * __restrict__ s_sorted_keys, const int ns, 
                                 const int * __restrict__ partition_starts, 
                                 key_t * __restrict__ merged_keys, 
                                 int * __restrict__ keys_idx, 
                                 int * __restrict__ keys_r_arr) {
    // OPTIMIZATION: __restrict__ and possibly unroll loops
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int r_start = partition_starts[2 * threadId];
    int s_start = partition_starts[2 * threadId + 1];

    int r_end = nr;
    int s_end = ns;
    
    if (threadId != NUM_THREADS * NUM_BLOCKS - 1) {
        r_end = partition_starts[2 * threadId + 2];
        s_end = partition_starts[2 * threadId + 3];
    }

    // Merge two sorted sublists
    while (r_start < r_end && s_start < s_end) {
        key_t r_key = r_sorted_keys[r_start];
        key_t s_key = s_sorted_keys[s_start];
        bool pick_r = (r_key < s_key);
        merged_keys[r_start + s_start] = pick_r ? r_key : s_key;
        keys_idx[r_start + s_start]   = pick_r ? r_start : s_start;
        keys_r_arr[r_start + s_start] = pick_r ? 1 : 0;
        r_start += pick_r;
        s_start += (!pick_r);
    }
    // Copy the remainder
    while (r_start < r_end) {
        merged_keys[r_start + s_start] = r_sorted_keys[r_start];
        keys_idx[r_start + s_start]   = r_start;
        keys_r_arr[r_start + s_start] = 1;
        r_start++;
    }
    while (s_start < s_end) {
        merged_keys[r_start + s_start] = s_sorted_keys[s_start];
        keys_idx[r_start + s_start]   = s_start;
        keys_r_arr[r_start + s_start] = 0;
        s_start++;
    }
}

template<typename key_t>
__global__ void merge_partition_sizes(const int nr, const int ns, 
                                      int * __restrict__ partition_matches, 
                                      const key_t * __restrict__ merged_keys, 
                                      int partition_size) {
    // OPTIMIZATION: __restrict__ used
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_start = threadId * partition_size;
    int idx_end   = min((threadId + 1) * partition_size, nr + ns);
    int count = 0;

    if (idx_start == 0 && idx_end > 0) {
        // The very first segment: start comparison from idx_start+1
        for (int i = idx_start + 1; i < idx_end; i++) {
            if (merged_keys[i] == merged_keys[i - 1]) count++;
        }
    } else {
        for (int i = idx_start; i < idx_end; i++) {
            if (i > 0 && merged_keys[i] == merged_keys[i - 1]) count++;
        }
    }

    partition_matches[threadId] = count;
}

template<typename key_t>
__global__ void merge_join_keys(const int nr, const int ns, 
                                key_t * __restrict__ keys_out, 
                                int * __restrict__ r_out, 
                                int * __restrict__ s_out, 
                                const int * __restrict__ prefix_partition_matches, 
                                const key_t * __restrict__ merged_keys, 
                                const int * __restrict__ keys_idx, 
                                const int * __restrict__ keys_r_arr, 
                                int partition_size, int output_buffer_size) {
    // OPTIMIZATION: __restrict__, precompute prefix
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_start = threadId * partition_size;
    int idx_end   = min((threadId + 1) * partition_size, nr + ns);

    int prev_count = (threadId == 0) ? 0 : prefix_partition_matches[threadId - 1];
    int count = prev_count;

    for (int i = idx_start; i < idx_end; i++) {
        if (i > 0 && merged_keys[i] == merged_keys[i - 1]) {
            int pos = count % output_buffer_size;
            keys_out[pos] = merged_keys[i];
            if (keys_r_arr[i] == 1) {
                r_out[pos] = keys_idx[i];
                s_out[pos] = keys_idx[i - 1];
            } else {
                r_out[pos] = keys_idx[i - 1];
                s_out[pos] = keys_idx[i];
            }
            count++;
        }
    }
}

template<typename key_t>
void our_merge_path(key_t *r_sorted_keys, const int nr, 
                    key_t *s_sorted_keys, const int ns, 
                    key_t *keys_out, int *r_out, int *s_out, 
                    int *num_matches, int output_buffer_size,
                    key_t *merged_keys, int *keys_idx, int *keys_r_arr) {
    std::cout << "Inside the custom merge path code\n";

    // Set cache preference for these kernels to prefer L1 (optional)
    cudaFuncSetCacheConfig(create_merge_partitions<key_t>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(sequential_merge<key_t>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(merge_partition_sizes<key_t>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(merge_join_keys<key_t>, cudaFuncCachePreferL1);

    int partition_size = (nr + ns + NUM_THREADS * NUM_BLOCKS - 1) / (NUM_THREADS * NUM_BLOCKS);
    int *partition_starts;
    int *partition_matches;

    allocate_mem(&partition_starts, false, sizeof(int) * 2 * NUM_THREADS * NUM_BLOCKS);
    allocate_mem(&partition_matches, false, sizeof(int) * NUM_THREADS * NUM_BLOCKS);

    // Create partitions
    create_merge_partitions<<<NUM_BLOCKS, NUM_THREADS>>>(r_sorted_keys, nr, s_sorted_keys, ns, partition_starts, partition_size);
    // Merge each partition sequentially (in parallel)
    sequential_merge<<<NUM_BLOCKS, NUM_THREADS>>>(r_sorted_keys, nr, s_sorted_keys, ns, partition_starts, merged_keys, keys_idx, keys_r_arr);
    // Compute how many matches in each partition
    merge_partition_sizes<<<NUM_BLOCKS, NUM_THREADS>>>(nr, ns, partition_matches, merged_keys, partition_size);

    // Inclusive scan to get prefix sums of matches
    thrust::device_ptr<int> d_partition_matches(partition_matches);
    thrust::inclusive_scan(d_partition_matches, d_partition_matches + (NUM_THREADS * NUM_BLOCKS), d_partition_matches);

    int final_sum;
    cudaMemcpy(&final_sum, partition_matches + (NUM_THREADS * NUM_BLOCKS - 1), sizeof(int), cudaMemcpyDeviceToHost);
    *num_matches = final_sum;

    // Materialize the final joined keys and indices
    merge_join_keys<<<NUM_BLOCKS, NUM_THREADS>>>(nr, ns, keys_out, r_out, s_out, partition_matches, merged_keys, keys_idx, keys_r_arr, partition_size, output_buffer_size);

    release_mem(partition_starts);
    release_mem(partition_matches);
}
