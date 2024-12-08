#include <cuda_runtime.h>
#include <cassert>
#include <cooperative_groups.h>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#define LOW_BIT 0
#define HIGH_BIT 4

template <
    typename KeyT, 
    typename ValueT, 
    typename CountT, 
    typename BucketT>
class Partition{
public:
    int size;
    KeyT *key_array;

    Partition(int size, KeyT *key_array) {
        this->size = size;
        this->key_array = key_array;
    }

    template <int low_bit, int high_bit>
    __host__ size_t get_workspace_size(dim3 grid_dim, dim3 block_dim) {
        assert(block_dim.y == 1 && block_dim.z == 1); // We only support 1D blocks for now
        assert(grid_dim.y == 1 && grid_dim.z == 1); // We only support 1D grids for now
        auto NUM_BUCKETS = 1 << (high_bit - low_bit);
        auto NUM_BLOCKS = grid_dim.x;
        // We need two arrays - one for block space and one for global block scan
        return NUM_BLOCKS * NUM_BUCKETS * sizeof(CountT) + NUM_BUCKETS * sizeof(CountT);
    }

    template <int low_bit, int high_bit>
    __host__ size_t get_shmem_size(dim3 grid_dim, dim3 block_dim) {
        assert(block_dim.y == 1 && block_dim.z == 1); // We only support 1D blocks for now
        assert(grid_dim.y == 1 && grid_dim.z == 1); // We only support 1D grids for now
        auto NUM_BUCKETS = 1 << (high_bit - low_bit);
        // We need two arrays - one for block space and one for global block scan
        return NUM_BUCKETS * sizeof(CountT);
    }

    template <int low_bit, int high_bit>
    __device__ BucketT get_bucket(KeyT key) {
        return key >> low_bit & ((1 << (high_bit - low_bit)) - 1);
    }
};

// CountT must support atomic writes
template<
    typename KeyT,
    typename ValueT,
    typename CountT,
    typename BucketT,
    int NUM_THREADS, 
    int THREAD_TILE, 
    int low_bit, 
    int high_bit>
__global__ void partition(Partition<KeyT, ValueT, CountT, BucketT> &p, void* workspace, void* out, void* debug_out) {
    assert(gridDim.y == 1 && gridDim.z == 1); // We only support 1D grids for now
    assert(blockDim.y == 1 && blockDim.z == 1); // We only support 1D blocks for now
    auto BLOCK_TILE = NUM_THREADS * THREAD_TILE;
    auto NUM_BUCKETS = 1 << (high_bit - low_bit);

    // Block space: We put status flags in top two bits
    CountT* block_space = ((CountT*) workspace) + blockIdx.x * NUM_BUCKETS;
    CountT* global_block_scan = &block_space[NUM_BUCKETS * gridDim.x];
    KeyT* output = (KeyT*) out;

    CountT PREFIX_DONE = 1 << (sizeof(CountT) * 8 - 1); // Top bit set means prefix sum is done
    CountT AGG_DONE = 1 << (sizeof(CountT) * 8 - 2); // Second top bit set means aggregation is done

    // Shared memory
    extern __shared__ CountT local_hist[];

    // // Initialize local histogram
    for (auto i = threadIdx.x; i < NUM_BUCKETS; i += NUM_THREADS) {
        local_hist[i] = 0;
    }
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        CountT* debug_ptr = (CountT*) debug_out;
        assert(debug_ptr + NUM_BUCKETS <= local_hist || debug_ptr >= local_hist + NUM_BUCKETS);
        assert(global_block_scan + NUM_BUCKETS <= local_hist || global_block_scan >= local_hist + NUM_BUCKETS);
        for(int i = 0; i < NUM_BUCKETS; i++) {
            debug_ptr[i] = global_block_scan[i];
            printf("global_block_scan[%d] = %d\n", i, global_block_scan[i]);
            printf("debug_out[%d] = %d\n", i, debug_ptr[i]);
        }
    }

    // Compute local histogram
    for (auto i = blockIdx.x * BLOCK_TILE + threadIdx.x; i < (blockIdx.x + 1) * BLOCK_TILE && i < p.size; i += NUM_THREADS) {
        BucketT key = p.template get_bucket<low_bit, high_bit>(p.key_array[i]);
        assert(key < NUM_BUCKETS);
        // uncoalesced writes: Optimize this later with warp shuffle
        // atomicAdd(&local_hist[key], 1);
    }
    __syncthreads();

    CountT write_status = (blockIdx.x == 0)? PREFIX_DONE: AGG_DONE;

    for (auto i = threadIdx.x; i < NUM_BUCKETS; i += NUM_THREADS) {
        // We should not have any status flags set in local_hist
        // assert(local_hist[i] & (PREFIX_DONE | AGG_DONE) == 0);
        // auto write_data = write_status | local_hist[i];
        // Assume writes are atomic - value and status are written together
        assert(&block_space[i] < global_block_scan);
        // block_space[i] = write_data;
    }

    // Each thread is responsible for some number of buckets
    // It performs a decoupled lookback to get the prefix sum for its buckets
    for (auto cur_bucket = threadIdx.x; cur_bucket < NUM_BUCKETS; cur_bucket += NUM_THREADS) {
        int prev_block = (int) blockIdx.x - 1;
        CountT local_val = local_hist[cur_bucket]; // No status flags set
        while (prev_block >= 0) {
            auto prev_space = ((CountT*) workspace) + prev_block * NUM_BUCKETS;
            auto prev_status = prev_space[cur_bucket];
            if (prev_status & PREFIX_DONE) {
                local_val += prev_status & ~(PREFIX_DONE | AGG_DONE); // Clear status flags
                break;
            } else if (prev_status & AGG_DONE) {
                local_val += prev_status & ~(PREFIX_DONE | AGG_DONE); // Clear status flags
                prev_block--;
            } else {
                // Keep spinning
            }
        }
        // // We should not have any status flags set in local_val
        // assert(local_val & (PREFIX_DONE | AGG_DONE) == 0);
        // // No race here - write to global
        // // Set status flag to PREFIX_DONE
        assert(&block_space[cur_bucket] < global_block_scan);
        block_space[cur_bucket] = local_val | PREFIX_DONE;
    }
    if (blockIdx.x == gridDim.x - 1) {
        // Last thread is responsible for bucket aggregation
        // Simple code for now - we know that thread x has the prefix sum for its buckets
        for (auto cur_bucket = threadIdx.x; cur_bucket < NUM_BUCKETS; cur_bucket += NUM_THREADS) {
            for (auto j = cur_bucket; j < NUM_BUCKETS; j++) {
                assert(&block_space[j] < global_block_scan);
                atomicAdd(&global_block_scan[cur_bucket], block_space[j] & ~(PREFIX_DONE | AGG_DONE));
            }
        }
    }
    cooperative_groups::thread_group g = cooperative_groups::this_thread_block();
    g.sync();
    // Implicit groups?
    // Now we have the global prefix sum

    // Now write out the data - each thread is responsible for a tile
    for (auto i = blockIdx.x * BLOCK_TILE; i < (blockIdx.x + 1) * BLOCK_TILE && i < p.size; i++) {
        BucketT bucket = p.template get_bucket<low_bit, high_bit>(p.key_array[i]);
        // Check if bucket belongs to this thread
        if (bucket % blockDim.x != threadIdx.x) {
            continue;
        }
        // Bucket internal offset
        auto local_offset = block_space[bucket] & ~(PREFIX_DONE | AGG_DONE);
        // Offset from previous buckets - inclusive scan
        auto global_offset = bucket == 0? 0: global_block_scan[bucket - 1];
        auto out_index = global_offset + local_offset;
        // No race - each thread updates its own buckets
        // output[--out_index] = p.key_array[i]; // Reverse order!!
    }
}

void test_1(){
    using KeyT = uint32_t;
    using ValueT = uint32_t;
    using CountT = uint32_t;
    using BucketT = uint32_t;

    auto NUM_BUCKETS = 1 << (HIGH_BIT - LOW_BIT);

    int size = 1;
    uint32_t *key_array = nullptr;
    uint32_t *host_array = new uint32_t[size];
    for(int i = 0; i < size; i++) {
        host_array[i] = rand() % NUM_BUCKETS; // Random values for testing
    }

    cudaMalloc(&key_array, size * sizeof(uint32_t));
    cudaMemcpy(key_array, host_array, size * sizeof(uint32_t), cudaMemcpyHostToDevice);

    Partition<KeyT, ValueT, CountT, BucketT> p(size, key_array);

    dim3 grid_dim((size + 31) / 32);
    dim3 block_dim(32);

    auto workspace_size = p.get_workspace_size<LOW_BIT, HIGH_BIT>(grid_dim, block_dim);
    auto shmem_size = p.get_shmem_size<LOW_BIT, HIGH_BIT>(grid_dim, block_dim);

    std::cout << "Workspace size: " << workspace_size << "\n";
    std::cout << "Shared memory size: " << shmem_size << "\n";

    void* workspace = nullptr;
    void* out = nullptr;
    void* debug_out = nullptr;

    cudaMalloc(&workspace, workspace_size);
    cudaMalloc(&out, size * sizeof(uint32_t));
    cudaMalloc(&debug_out, NUM_BUCKETS * sizeof(uint32_t));

    cudaMemset(workspace, 0, workspace_size);
    cudaMemset(debug_out, 0, NUM_BUCKETS * sizeof(uint32_t));
    cudaDeviceSynchronize();

    partition<KeyT, ValueT, CountT, BucketT, 32, 1, LOW_BIT, HIGH_BIT><<<grid_dim, block_dim, shmem_size>>>(p, workspace, out, debug_out);
    cudaDeviceSynchronize();

    uint32_t* debug_host = new uint32_t[NUM_BUCKETS];
    cudaMemcpy(debug_host, debug_out, NUM_BUCKETS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for(int i = 0; i < NUM_BUCKETS; i++) {
        std::cout << debug_host[i] << "\n";
    }

    // uint32_t* output_host = new uint32_t[size];
    // cudaMemcpy(output_host, out, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < size; i++) {
    //     std::cout << output_host[i] << "\n";
    // }

    // Cleanup
    cudaFree(workspace);
    cudaFree(key_array);
    cudaFree(out);
    cudaFree(debug_out);
    delete[] host_array;
    delete[] debug_host;
    // delete[] output_host;
}



int main() {
    test_1();
}