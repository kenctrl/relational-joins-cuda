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


#define MP_NUM_THREADS 128
#define MP_NUM_BLOCKS 48

template<typename key_t>
__global__ void create_merge_partitions_kernel(
    const key_t *r_sorted_keys, int nr,
    const key_t *s_sorted_keys, int ns,
    int partition_size, int *partition_starts, int total_partitions)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= total_partitions) return;

    int diag = threadId * partition_size;
    if (diag > nr + ns) {
        partition_starts[2*threadId] = nr;
        partition_starts[2*threadId+1] = ns;
        return;
    }

    int r_low = max(0, diag - ns);
    int r_high = min(diag, nr);

    while (r_low < r_high) {
        int r_mid = (r_low + r_high) >> 1;
        int s_mid = diag - r_mid;

        key_t a_key = (r_mid < nr) ? r_sorted_keys[r_mid] : (key_t)INT_MAX;
        key_t b_key = (s_mid > 0) ? s_sorted_keys[s_mid - 1] : (key_t)(-INT_MAX);

        if (s_mid > 0 && a_key < b_key) {
            r_low = r_mid + 1;
        } else {
            r_high = r_mid;
        }
    }

    int s_low = diag - r_low;
    partition_starts[2 * threadId] = r_low;
    partition_starts[2 * threadId + 1] = s_low;
}

// Kernel to count matches only
template<int NT=128, typename key_t>
__global__ void count_matches_kernel(
    const key_t* r_sorted_keys, int nr,
    const key_t* s_sorted_keys, int ns,
    const int* partition_starts, int partition_size,
    int *partition_matches,
    int total_partitions)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_partitions) return;

    int r_start = partition_starts[2 * tid];
    int s_start = partition_starts[2 * tid + 1];
    int r_end = (tid == total_partitions - 1) ? nr : partition_starts[2 * tid + 2];
    int s_end = (tid == total_partitions - 1) ? ns : partition_starts[2 * tid + 3];

    int r_idx = r_start;
    int s_idx = s_start;

    int match_count = 0;

    // Merge and count matches without output
    while (r_idx < r_end && s_idx < s_end) {
        key_t rk = r_sorted_keys[r_idx];
        key_t sk = s_sorted_keys[s_idx];
        if (rk == sk) {
            match_count++;
            r_idx++;
            s_idx++;
        } else if (rk < sk) {
            r_idx++;
        } else {
            s_idx++;
        }
    }

    partition_matches[tid] = match_count;
}

// Kernel to write matches using prefix sums
template<int NT=128, typename key_t>
__global__ void write_matches_kernel(
    const key_t* r_sorted_keys, int nr,
    const key_t* s_sorted_keys, int ns,
    const int* partition_starts, int partition_size,
    const int *prefix_partition_matches,
    key_t* keys_out, int* r_out, int* s_out,
    int output_buffer_size,
    int total_partitions)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_partitions) return;

    int r_start = partition_starts[2 * tid];
    int s_start = partition_starts[2 * tid + 1];
    int r_end = (tid == total_partitions - 1) ? nr : partition_starts[2 * tid + 2];
    int s_end = (tid == total_partitions - 1) ? ns : partition_starts[2 * tid + 3];

    int r_idx = r_start;
    int s_idx = s_start;

    // offset for this partition's matches
    int offset = (tid == 0) ? 0 : prefix_partition_matches[tid - 1];

    // Merge again and write matches
    while (r_idx < r_end && s_idx < s_end) {
        key_t rk = r_sorted_keys[r_idx];
        key_t sk = s_sorted_keys[s_idx];
        if (rk == sk) {
            int pos = offset++;
            int w_pos = pos % output_buffer_size;
            keys_out[w_pos] = rk;
            r_out[w_pos] = r_idx;
            s_out[w_pos] = s_idx;
            r_idx++;
            s_idx++;
        } else if (rk < sk) {
            r_idx++;
        } else {
            s_idx++;
        }
    }
}

template<typename key_t>
void our_merge_path(
    key_t *r_sorted_keys, const int nr,
    key_t *s_sorted_keys, const int ns,
    key_t *keys_out, int *r_out, int *s_out,
    int *num_matches, int output_buffer_size)
{
    int total_threads = MP_NUM_THREADS * MP_NUM_BLOCKS;
    int total_partitions = total_threads;
    int partition_size = (nr + ns + total_threads - 1) / total_threads;

    int *partition_starts;
    allocate_mem(&partition_starts, false, sizeof(int)*2*total_partitions);

    // compute partitions
    create_merge_partitions_kernel<<<MP_NUM_BLOCKS, MP_NUM_THREADS>>>(
        r_sorted_keys, nr,
        s_sorted_keys, ns,
        partition_size, partition_starts, total_partitions
    );
    // CHECK_ERROR(cudaGetLastError());

    // count matches only
    int* partition_matches;
    allocate_mem(&partition_matches, false, sizeof(int)*total_partitions);

    count_matches_kernel<<<MP_NUM_BLOCKS, MP_NUM_THREADS>>>(
        r_sorted_keys, nr,
        s_sorted_keys, ns,
        partition_starts, partition_size,
        partition_matches,
        total_partitions
    );
    // CHECK_ERROR(cudaGetLastError());

    // prefix sum of matches
    thrust::device_ptr<int> d_partition_matches(partition_matches);
    thrust::inclusive_scan(d_partition_matches, d_partition_matches + total_partitions, d_partition_matches);

    int final_count;
    cudaMemcpy(&final_count, partition_matches + (total_partitions - 1), sizeof(int), cudaMemcpyDeviceToHost);
    *num_matches = final_count;

    // write matches using prefix sums
    write_matches_kernel<<<MP_NUM_BLOCKS, MP_NUM_THREADS>>>(
        r_sorted_keys, nr,
        s_sorted_keys, ns,
        partition_starts, partition_size,
        partition_matches,
        keys_out, r_out, s_out,
        output_buffer_size,
        total_partitions
    );
    // CHECK_ERROR(cudaGetLastError());

    release_mem(partition_starts);
    release_mem(partition_matches);
}