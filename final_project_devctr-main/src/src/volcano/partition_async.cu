#include <cuda_runtime.h>
#include <cassert>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#define LOW_BIT 0
#define HIGH_BIT 8

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
    return (NUM_BLOCKS * NUM_BUCKETS * sizeof(CountT) + NUM_BUCKETS * sizeof(CountT)) * 2;
}

template <typename KeyT, typename CountT, int THREAD_TILE, int low_bit, int high_bit>
__host__ size_t get_shmem_size(dim3 grid_dim, dim3 block_dim) {
    // assert(block_dim.y == 1 && block_dim.z == 1); // We only support 1D blocks for now
    // assert(grid_dim.y == 1 && grid_dim.z == 1); // We only support 1D grids for now
    auto NUM_BUCKETS = 1 << (high_bit - low_bit);
    // We need a local histogram
    return (NUM_BUCKETS * sizeof(CountT) + THREAD_TILE * block_dim.x * sizeof(KeyT)) * 2;
}

template <typename BucketT, typename KeyT, int low_bit, int high_bit>
__device__ BucketT get_bucket(KeyT key) {
    return (key >> low_bit) & ((1 << (high_bit - low_bit)) - 1);
}

// CountT must support atomic writes
template<
    typename KeyT,
    typename ValueT,
    typename CountT,
    typename BucketT,
    int NUM_THREADS, 
    int THREAD_TILE, 
    int LOOP_TILE,
    int low_bit, 
    int high_bit>
__global__ void partition(KeyT* key_array, size_t size, void* workspace, void* out, void* debug_out) {
    CountT* debug_ptr = (CountT*) debug_out;

    auto BLOCK_TILE = NUM_THREADS * THREAD_TILE;
    auto NUM_BUCKETS = 1 << (high_bit - low_bit);

    // Block space: We put status flags in top two bits
    // Block space points to current block's histogram in workspace
    CountT* block_space0 = static_cast<CountT*>(workspace) + blockIdx.x * NUM_BUCKETS;
    CountT* block_space1 = block_space0 + NUM_BUCKETS * gridDim.x;
    // Global scan array starts after all block histograms
    CountT* global_block_scan0 = static_cast<CountT*>(workspace) + 2 * gridDim.x * NUM_BUCKETS;
    CountT* global_block_scan1 = global_block_scan0 + NUM_BUCKETS;

    CountT* block_space[2] = {block_space0, block_space1};
    CountT* global_block_scans[2] = {global_block_scan0, global_block_scan1};

    CountT PREFIX_DONE = CountT(1) << (sizeof(CountT) * 8 - 1); // Top bit set means prefix sum is done
    CountT AGG_DONE = CountT(1) << (sizeof(CountT) * 8 - 2); // Second top bit set means aggregation is done

    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    // Shared memory
    extern __shared__ int cache[];

    CountT* local_hist0 = (CountT*) cache;
    CountT* local_hist1 = &local_hist0[NUM_BUCKETS];

    KeyT* local_keys0 = (KeyT*) &local_hist1[NUM_BUCKETS];
    KeyT* local_keys1 = &local_keys0[BLOCK_TILE];

    KeyT* keys_buffer[2] = {local_keys0, local_keys1};
    CountT* hist_buffer[2] = {local_hist0, local_hist1};
    KeyT* output = (KeyT*) out;

    int stage = 0;

    auto LOOP_STRIDE = gridDim.x * BLOCK_TILE;

    // First memory copy
    cooperative_groups::memcpy_async(block, keys_buffer[stage], key_array + blockIdx.x * BLOCK_TILE, BLOCK_TILE * sizeof(KeyT));
    key_array += LOOP_STRIDE;

    // Initialize local histogram
    for (auto i = threadIdx.x; i < NUM_BUCKETS; i += NUM_THREADS) {
        hist_buffer[stage][i] = 0;
    }
    // Kick off next stage
    cooperative_groups::memcpy_async(block, keys_buffer[stage ^ 1], key_array + blockIdx.x * BLOCK_TILE, BLOCK_TILE * sizeof(KeyT));
    key_array += LOOP_STRIDE;
    cooperative_groups::wait_prior<1>(block); // sync + memcopy
    // Compute local histogram
    for (auto i = threadIdx.x; i < BLOCK_TILE; i += NUM_THREADS) {
        BucketT key = get_bucket<BucketT, KeyT, low_bit, high_bit>(keys_buffer[stage][i]);
        // // uncoalesced writes: Optimize this later with warp shuffle
        atomicAdd(&hist_buffer[stage][key], 1);
    };

    // Main loop
    for (auto outer_loop = 0; outer_loop < LOOP_TILE; outer_loop++){
        CountT write_status = (blockIdx.x == 0)? PREFIX_DONE: AGG_DONE;
        __syncthreads();
        for (auto i = threadIdx.x; i < NUM_BUCKETS; i += NUM_THREADS) {
            auto write_data = write_status | hist_buffer[stage][i];
            block_space[stage][i] = write_data;
        }

        if (threadIdx.x == 0) {
            for (auto i = 0; i < NUM_BUCKETS; i++) {
                assert(block_space[stage][i] & (PREFIX_DONE | AGG_DONE));
            }
        }

        // Each thread is responsible for some number of buckets
        // It performs a decoupled lookback to get the prefix sum for its buckets
        for (auto cur_bucket = threadIdx.x; cur_bucket < NUM_BUCKETS; cur_bucket += NUM_THREADS) {
            int prev_block = (int) blockIdx.x - 1;
            CountT local_val = hist_buffer[stage][cur_bucket]; // No status flags set
            while (prev_block >= 0) {
                auto prev_space = ((CountT*) workspace) + prev_block * NUM_BUCKETS + stage * NUM_BUCKETS * gridDim.x;
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
            block_space[stage][cur_bucket] = local_val | PREFIX_DONE;
            assert(block_space[stage][cur_bucket] & PREFIX_DONE);
        }
        if (blockIdx.x == gridDim.x - 1) {
            // Simple code for now - we know that thread x has the prefix sum for its buckets
            for (auto cur_bucket = threadIdx.x; cur_bucket < NUM_BUCKETS; cur_bucket += NUM_THREADS) {
                auto thread_val = block_space[stage][cur_bucket] & ~(PREFIX_DONE | AGG_DONE);
                for (auto j = cur_bucket; j < NUM_BUCKETS; j++) {
                    atomicAdd(&global_block_scans[stage][j], thread_val);
                }
                // Set the next global_block_scan to 0 for the next iteration
                global_block_scans[stage ^ 1][cur_bucket] = 0;

            }
        }
        // Initialize next local histogram
        for (auto i = threadIdx.x; i < NUM_BUCKETS; i += NUM_THREADS) {
            hist_buffer[stage ^ 1][i] = 0;
        }
        cooperative_groups::wait_prior<0>(block); // sync + memcopy
        // Compute next local histogram
        for (auto i = threadIdx.x; i < BLOCK_TILE; i += NUM_THREADS) {
            BucketT key = get_bucket<BucketT, KeyT, low_bit, high_bit>(keys_buffer[stage ^ 1][i]);
            // // uncoalesced writes: Optimize this later with warp shuffle
            atomicAdd(&hist_buffer[stage ^ 1][key], 1);
        };

        // Remove enabled status flags
        for (auto i = threadIdx.x; i < NUM_BUCKETS; i += NUM_THREADS) {
            block_space[stage ^ 1][i] = 0;
        }

        grid.sync();
        __threadfence(); // Ensure all writes are visible

        // Now we have the global prefix sum
        // Now write out the data - each thread is responsible for a tile
        for (auto i = 0; i < BLOCK_TILE; i++) {
            BucketT bucket = get_bucket<BucketT, KeyT, low_bit, high_bit>(keys_buffer[stage][i]);
            // Check if bucket belongs to this thread
            if (bucket % NUM_THREADS != threadIdx.x) {
                continue;
            }
            // Bucket internal offset, remove status flags
            auto local_offset = (block_space[stage][bucket] - hist_buffer[stage][bucket]--) & ~(PREFIX_DONE | AGG_DONE);
            // Offset from previous buckets - inclusive scan
            auto global_offset = bucket > 0? global_block_scans[stage][bucket - 1]: 0;
            auto out_index = global_offset + local_offset;
            // No race - each thread updates its own buckets
            output[out_index] = keys_buffer[stage][i];
        }
        output += LOOP_STRIDE;
        // Swap buffers
        stage ^= 1;
    }
    // Kick off next stage
    cooperative_groups::memcpy_async(block, keys_buffer[stage], key_array + blockIdx.x * BLOCK_TILE, BLOCK_TILE * sizeof(KeyT));
    key_array += LOOP_STRIDE;
}


void test_1(){
    using KeyT = uint32_t;
    using ValueT = uint32_t;
    using CountT = uint32_t;
    using BucketT = uint32_t;

    auto NUM_BUCKETS = 1 << (HIGH_BIT - LOW_BIT);

    constexpr int NUM_THREADS = 64;
    constexpr int THREAD_TILE = 8;
    constexpr int LOOP_TILE = 4;

    size_t size = 1 << 13;
    uint32_t *key_array = nullptr; // device memory
    uint32_t *host_array = new uint32_t[size]; // host memory
    for(auto i = 0; i < size; i++) {
        // host_array[i] = rand() % NUM_BUCKETS; // Random values for testing
        host_array[i] = size - i - 1; // Reverse order
    }

    cudaMalloc(&key_array, size * sizeof(KeyT));
    cudaMemcpy(key_array, host_array, size * sizeof(uint32_t), cudaMemcpyHostToDevice);

    dim3 grid_dim((size + NUM_THREADS * THREAD_TILE * LOOP_TILE - 1) / (NUM_THREADS * THREAD_TILE * LOOP_TILE));
    dim3 block_dim(NUM_THREADS);

    auto workspace_size = get_workspace_size<CountT, LOW_BIT, HIGH_BIT>(grid_dim, block_dim);
    auto shmem_size = get_shmem_size<CountT, KeyT, THREAD_TILE, LOW_BIT, HIGH_BIT>(grid_dim, block_dim);

    std::cout << "Workspace size: " << workspace_size << "\n";
    std::cout << "Shared memory size: " << shmem_size << "\n";
    std::cout << "Workload: " << size << "\n";

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
        partition<KeyT, ValueT, CountT, BucketT, NUM_THREADS, THREAD_TILE, LOOP_TILE, LOW_BIT, HIGH_BIT>, 
        NUM_THREADS, shmem_size);
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    // assert(numBlocksPerSm * deviceProp.multiProcessorCount >= grid_dim.x);

    std::cout << "Blocks per SM: " << numBlocksPerSm << "\n";
    std::cout << "Max Blocks: " << numBlocksPerSm * deviceProp.multiProcessorCount << "\n";
    std::cout << "Grid dim: " << grid_dim.x << "\n";
    std::cout << "Cooperative launch: " << supportsCoopLaunch << "\n";
    // assert(supportsCoopLaunch);

    void* args[] = {
        (void*)&key_array,
        (void*)&size,
        (void*)&workspace,
        (void*)&out,
        (void*)&debug_out
    };
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(partition<KeyT, ValueT, CountT, BucketT, NUM_THREADS, THREAD_TILE, LOOP_TILE, LOW_BIT, HIGH_BIT>),
        grid_dim,
        block_dim,
        args,
        shmem_size,
        nullptr
    );
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t* debug_host = new uint32_t[NUM_BUCKETS];
    cudaMemcpy(debug_host, debug_out, NUM_BUCKETS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < NUM_BUCKETS; i++) {
    //     std::cout << debug_host[i] << "\n";
    // }

    uint32_t* output_host = new uint32_t[size];
    cudaMemcpy(output_host, out, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < size; i++) {
    //     std::cout << output_host[i] << "\n";
    // }

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

template <typename KeyT, typename BucketT, int low_bit, int high_bit>
__global__ void key_tester(KeyT* key_array, BucketT* out_array, size_t size) {

    for (int i = 0; i < size; i++) {
        out_array[i] = get_bucket<BucketT, KeyT, low_bit, high_bit>(key_array[i]);
        assert(out_array[i] == key_array[i] % 256);
    }
}

void test_2(){
    using KeyT = uint32_t;
    using BucketT = uint32_t;

    constexpr int NUM_KEYS = 32;
    KeyT host_keys[NUM_KEYS];
    for(int i = 0; i < NUM_KEYS; i++) {
        host_keys[i] = i;
    }

    KeyT* cuda_keys;
    cudaMalloc(&cuda_keys, NUM_KEYS * sizeof(KeyT));
    cudaMemcpy(cuda_keys, host_keys, NUM_KEYS * sizeof(KeyT), cudaMemcpyHostToDevice);

    dim3 grid_dim(1);
    dim3 block_dim(1);

    BucketT* cuda_out;
    cudaMalloc(&cuda_out, NUM_KEYS * sizeof(BucketT));

    key_tester<KeyT, BucketT, LOW_BIT, HIGH_BIT><<<grid_dim, block_dim>>>(cuda_keys, cuda_out, NUM_KEYS);
}


int main() {
    test_1();
    // test_2();
}