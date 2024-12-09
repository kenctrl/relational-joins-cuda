#include <cuda_runtime.h>
#include <cassert>
#include <cooperative_groups.h>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#define LOW_BIT 0
#define HIGH_BIT 4

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

template <typename CountT, int low_bit, int high_bit>
__host__ size_t get_workspace_size(dim3 grid_dim, dim3 block_dim) {
    // assert(block_dim.y == 1 && block_dim.z == 1); // We only support 1D blocks for now
    // assert(grid_dim.y == 1 && grid_dim.z == 1); // We only support 1D grids for now
    auto NUM_BUCKETS = 1 << (high_bit - low_bit);
    auto NUM_BLOCKS = grid_dim.x;
    // We need two arrays - one for block space and one for global block scan
    return NUM_BLOCKS * NUM_BUCKETS * sizeof(CountT) + NUM_BUCKETS * sizeof(CountT);
}

template <typename CountT, int low_bit, int high_bit>
__host__ size_t get_shmem_size(dim3 grid_dim, dim3 block_dim) {
    // assert(block_dim.y == 1 && block_dim.z == 1); // We only support 1D blocks for now
    // assert(grid_dim.y == 1 && grid_dim.z == 1); // We only support 1D grids for now
    auto NUM_BUCKETS = 1 << (high_bit - low_bit);
    // We need a local histogram
    return NUM_BUCKETS * sizeof(CountT);
}

template <typename BucketT, typename KeyT, int low_bit, int high_bit>
__device__ BucketT get_bucket(KeyT key) {
    return key >> low_bit & ((1 << (high_bit - low_bit)) - 1);
}

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
__global__ void partition(KeyT* key_array, int size, void* workspace, void* out, void* debug_out) {
    CountT* debug_ptr = (CountT*) debug_out;

    // assert(gridDim.y == 1 && gridDim.z == 1); // We only support 1D grids for now
    // assert(blockDim.y == 1 && blockDim.z == 1); // We only support 1D blocks for now
    // assert(blockDim.x == NUM_THREADS); // We only support 1D blocks for now

    auto BLOCK_TILE = NUM_THREADS * THREAD_TILE;
    auto NUM_BUCKETS = 1 << (high_bit - low_bit);

    // Block space: We put status flags in top two bits
    // Block space points to current block's histogram in workspace
    CountT* block_space = static_cast<CountT*>(workspace) + blockIdx.x * NUM_BUCKETS;
    // Global scan array starts after all block histograms
    CountT* global_block_scan = static_cast<CountT*>(workspace) + gridDim.x * NUM_BUCKETS;
    KeyT* output = (KeyT*) out;

    // Assert that memory is not overlapping

    CountT PREFIX_DONE = CountT(1) << (sizeof(CountT) * 8 - 1); // Top bit set means prefix sum is done
    CountT AGG_DONE = CountT(1) << (sizeof(CountT) * 8 - 2); // Second top bit set means aggregation is done
    CountT ALL_DONE = PREFIX_DONE | AGG_DONE;

    // assert(__popc(PREFIX_DONE) == 1);
    // assert(__popc(AGG_DONE) == 1);
    // assert(__popc(ALL_DONE) == 2);

    // Shared memory
    extern __shared__ CountT local_hist[];

    // // Initialize local histogram
    for (auto i = threadIdx.x; i < NUM_BUCKETS; i += NUM_THREADS) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Compute local histogram
    for (auto i = blockIdx.x * BLOCK_TILE + threadIdx.x; i < (blockIdx.x + 1) * BLOCK_TILE && i < size; i += NUM_THREADS) {
        BucketT key = get_bucket<BucketT, KeyT, low_bit, high_bit>(key_array[i]);
        // // uncoalesced writes: Optimize this later with warp shuffle
        // assert(key < NUM_BUCKETS);
        atomicAdd(&local_hist[key], 1);
    };
    __syncthreads();

    CountT write_status = (blockIdx.x == 0)? PREFIX_DONE: AGG_DONE;

    for (auto i = threadIdx.x; i < NUM_BUCKETS; i += NUM_THREADS) {
        // We should not have any status flags set in local_hist
        // assert((local_hist[i] & (PREFIX_DONE | AGG_DONE)) == 0);
        auto write_data = write_status | local_hist[i];
        // Assume writes are atomic - value and status are written together
        block_space[i] = write_data;
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
        // We should not have any status flags set in local_val
        // assert((local_val & (PREFIX_DONE | AGG_DONE)) == 0);
        // No race here - write to global
        // Set status flag to PREFIX_DONE
        block_space[cur_bucket] = local_val | PREFIX_DONE;
    }
    if (blockIdx.x == gridDim.x - 1) {
        // Simple code for now - we know that thread x has the prefix sum for its buckets
        for (auto cur_bucket = threadIdx.x; cur_bucket < NUM_BUCKETS; cur_bucket += NUM_THREADS) {
            auto thread_val = block_space[cur_bucket] & ~(PREFIX_DONE | AGG_DONE);
            for (auto j = cur_bucket; j < NUM_BUCKETS; j++) {
                atomicAdd(&global_block_scan[j], thread_val);
            }
        }
    }
    cooperative_groups::this_grid().sync();
    __threadfence(); // Ensure all writes are visible
    // // Implicit groups?

    // // Now we have the global prefix sum
    // Now write out the data - each thread is responsible for a tile
    for (auto i = blockIdx.x * BLOCK_TILE; i < (blockIdx.x + 1) * BLOCK_TILE && i < size; i++) {
        BucketT bucket = get_bucket<BucketT, KeyT, low_bit, high_bit>(key_array[i]);
        // Check if bucket belongs to this thread
        if (bucket % blockDim.x != threadIdx.x) {
            continue;
        }
        // Bucket internal offset, remove status flags
        auto local_offset = (--block_space[bucket]) & ~(PREFIX_DONE | AGG_DONE);
        // Offset from previous buckets - inclusive scan
        volatile auto global_offset = bucket > 0? global_block_scan[bucket - 1]: 0;
        // Clear status flags
        auto out_index = global_offset + local_offset;
        // No race - each thread updates its own buckets
        // assert(out_index < size && out_index >= 0);
        output[out_index] = key_array[i]; // Reverse order!!
    }
}


void test_1(){
    using KeyT = uint32_t;
    using ValueT = uint32_t;
    using CountT = uint32_t;
    using BucketT = uint32_t;

    constexpr int NUM_THREADS = 32;
    constexpr int THREAD_TILE = 1;

    auto NUM_BUCKETS = 1 << (HIGH_BIT - LOW_BIT);

    int size = 45;
    uint32_t *key_array = nullptr; // device memory
    uint32_t *host_array = new uint32_t[size]; // host memory
    for(int i = 0; i < size; i++) {
        // host_array[i] = rand() % NUM_BUCKETS; // Random values for testing
        host_array[i] = size - i - 1; // Reverse order
    }

    cudaMalloc(&key_array, size * sizeof(KeyT));
    cudaMemcpy(key_array, host_array, size * sizeof(uint32_t), cudaMemcpyHostToDevice);

    dim3 grid_dim((size + NUM_THREADS * THREAD_TILE - 1) / (NUM_THREADS * THREAD_TILE));
    dim3 block_dim(NUM_THREADS);

    auto workspace_size = get_workspace_size<CountT, LOW_BIT, HIGH_BIT>(grid_dim, block_dim);
    auto shmem_size = get_shmem_size<CountT, LOW_BIT, HIGH_BIT>(grid_dim, block_dim);

    std::cout << "Workspace size: " << workspace_size << "\n";
    std::cout << "Shared memory size: " << shmem_size << "\n";

    void* workspace = nullptr;
    void* out = nullptr;
    void* debug_out = nullptr;

    cudaMalloc(&workspace, workspace_size);
    cudaMalloc(&out, size * sizeof(KeyT));
    cudaMalloc(&debug_out, NUM_BUCKETS * sizeof(CountT));

    cudaMemset(workspace, 0, workspace_size);
    cudaMemset(debug_out, 0, NUM_BUCKETS * sizeof(CountT));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaDeviceProp deviceProp;
    int dev = 0;
    int numBlocksPerSm = 0;
    int supportsCoopLaunch = 0;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, 
        partition<KeyT, ValueT, CountT, BucketT, NUM_THREADS, THREAD_TILE, LOW_BIT, HIGH_BIT>, 
        NUM_THREADS, shmem_size);
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    // assert(numBlocksPerSm * deviceProp.multiProcessorCount >= grid_dim.x);

    std::cout << "Blocks per SM: " << numBlocksPerSm << "\n";
    std::cout << "Blocks: " << numBlocksPerSm * deviceProp.multiProcessorCount << "\n";
    std::cout << "Cooperative launch: " << supportsCoopLaunch << "\n";
    // assert(supportsCoopLaunch);

    void* args[] = {
        (void*)&key_array,
        (void*)&size,
        (void*)&workspace,
        (void*)&out,
        (void*)&debug_out
    };
    cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(partition<KeyT, ValueT, CountT, BucketT, 32, 1, LOW_BIT, HIGH_BIT>),
        grid_dim,
        block_dim,
        args,
        shmem_size,
        nullptr
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t* debug_host = new uint32_t[NUM_BUCKETS];
    cudaMemcpy(debug_host, debug_out, NUM_BUCKETS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < NUM_BUCKETS; i++) {
    //     std::cout << debug_host[i] << "\n";
    // }

    uint32_t* output_host = new uint32_t[size];
    cudaMemcpy(output_host, out, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        std::cout << output_host[i] << "\n";
    }

    // for (int i = 0; i < size; i++) {
    //     std::cout << host_array[i] << "\n";
    // }

    // Cleanup
    cudaFree(workspace);
    cudaFree(key_array);
    cudaFree(out);
    cudaFree(debug_out);
    delete[] host_array;
    delete[] debug_host;
    delete[] output_host;
}



int main() {
    test_1();
}