#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#define WARPSIZE 32

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

template <typename KeyT>
__device__ __forceinline__ int diag_search(
    KeyT* R,
    KeyT* S,
    const size_t r_size,
    const size_t s_size,
    int diag
) {
    if (diag < 0) return 0;
    // first element in R that is greater than S[diag - i - 1]
    auto lo = max(0, static_cast<int>(diag) - static_cast<int>(s_size));
    auto hi = min(static_cast<int>(diag), static_cast<int>(r_size));

    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (R[mid] <= S[diag - mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Only works for unique keys
template <
    typename KeyT,
    int THREAD_TILE,
    int NUM_THREADS>
__global__ void merge_path(
    KeyT* R,
    KeyT* S,
    const size_t r_size,
    const size_t s_size,
    KeyT* out,
    void* workspace
){
    constexpr int BLOCK_TILE = THREAD_TILE * NUM_THREADS;
    __shared__ KeyT shared_keys[(BLOCK_TILE + 1) * 2];

    auto shared_R = shared_keys;
    auto shared_S = shared_keys + (BLOCK_TILE + 1);

    auto diag_start = blockIdx.x * BLOCK_TILE * 2;
    auto diag_end = min(static_cast<int>(diag_start + BLOCK_TILE * 2), static_cast<int>(r_size + s_size));

    // clopen
    int block_R_start, block_R_end;
    // Two pass - can be optimized
    if (threadIdx.x == 0) {
        block_R_start = diag_search(R, S, r_size, s_size, diag_start - 1);
    } else if (threadIdx.x == NUM_THREADS - 1) {
        block_R_end = diag_search(R, S, r_size, s_size, diag_end - 1);
    }
    block_R_start = __shfl_sync(0xffffffff, block_R_start, 0);
    block_R_end = __shfl_sync(0xffffffff, block_R_end, NUM_THREADS - 1);

    auto block_S_start = diag_start - block_R_start;
    auto block_S_end = diag_end - block_R_end;

    // Total loads 2 * diag_diff
    for (auto idx_R = block_R_start + threadIdx.x; idx_R < block_R_end; idx_R += NUM_THREADS){
        auto local_idx = idx_R - block_R_start;
        shared_R[local_idx] = R[idx_R];
    }
    for (auto idx_S = block_S_start + threadIdx.x; idx_S < block_S_end; idx_S += NUM_THREADS){
        auto local_idx = idx_S - block_S_start;
        shared_S[local_idx] = S[idx_S];
    }
    if (threadIdx.x == 0) shared_S[block_S_end - block_S_start] = INT_MAX;
    if (threadIdx.x == NUM_THREADS - 1) shared_R[block_R_end - block_R_start] = INT_MAX;
    __syncthreads();

    auto thread_diag_start = threadIdx.x * THREAD_TILE * 2;
    auto thread_diag_end = min(static_cast<int>(thread_diag_start + 2 * THREAD_TILE), static_cast<int>(diag_end - diag_start));

    // Find start and end from shared memory
    auto block_size = (diag_end - diag_start) / 2;
    auto thread_R_start = diag_search(shared_R, shared_S, block_size, block_size, thread_diag_start - 1);

    auto idx_R = thread_R_start;
    auto idx_S = thread_diag_start - idx_R;
    auto diag = thread_diag_start;

    while (diag < thread_diag_end) {
        if (shared_R[idx_R] <= shared_S[idx_S]) {
            out[diag_start + diag] = shared_R[idx_R];
            idx_R++;
        } else {
            out[diag_start + diag] = shared_S[idx_S];
            idx_S++;
        }
        diag++;
    }
}

void visual_test_1(){
    // Initialize input arrays
    const size_t r_size = 5;
    const size_t s_size = 5;
    int h_R[r_size] = {1, 3, 5, 7, 9};
    int h_S[s_size] = {2, 4, 6, 8, 10};

    // Allocate device memory
    int *d_R, *d_S, *d_out;
    CUDA_CHECK(cudaMalloc(&d_R, r_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_S, s_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, (r_size + s_size) * sizeof(int)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_R, h_R, r_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S, h_S, s_size * sizeof(int), cudaMemcpyHostToDevice));

    // Define kernel parameters
    const int THREAD_TILE = 1;
    const int NUM_THREADS = 32; // Multiple of WARPSIZE
    const int num_blocks = (r_size + s_size + THREAD_TILE * NUM_THREADS - 1) / (THREAD_TILE * NUM_THREADS);

    // Launch kernel
    merge_path<int, THREAD_TILE, NUM_THREADS><<<num_blocks, NUM_THREADS>>>(
        d_R, d_S, r_size, s_size, d_out, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back to host
    int h_out[r_size + s_size];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (r_size + s_size) * sizeof(int), cudaMemcpyDeviceToHost));

    // Print output
    for (size_t i = 0; i < r_size + s_size; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    // Expected output: 1 2 3 4 5 6 7 8 9 10
    std::cout << "Expected output: 1 2 3 4 5 6 7 8 9 10" << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_out));
}
void correctness_test_2(){
    // Initialize input arrays
    constexpr int size = 32;

    const size_t r_size = size;
    const size_t s_size = size;

    int h_R[r_size];
    int h_S[s_size];

    int h_comp[r_size + s_size];

    constexpr int MAX_VAL = 1000;

    std::srand(std::time(nullptr));
    for (int idx = 0; idx < r_size; idx++) {
        h_R[idx] = std::rand() % MAX_VAL; // Random values between 0 and 99
    }
    for (int idx = 0; idx < s_size; idx++) {
        h_S[idx] = std::rand() % MAX_VAL; // Random values between 0 and 99
    }

    // Sort the arrays
    std::sort(h_R, h_R + r_size);
    std::sort(h_S, h_S + s_size);

    // Merge the arrays on the host
    int i = 0, j = 0, k = 0;
    while (i < r_size && j < s_size) {
        if (h_R[i] < h_S[j]) {
            h_comp[k++] = h_R[i++];
        } else {
            h_comp[k++] = h_S[j++];
        }
    }
    while (i < r_size) {
        h_comp[k++] = h_R[i++];
    }
    while (j < s_size) {
        h_comp[k++] = h_S[j++];
    }

    // Allocate device memory
    int *d_R, *d_S, *d_out;
    CUDA_CHECK(cudaMalloc(&d_R, r_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_S, s_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, (r_size + s_size) * sizeof(int)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_R, h_R, r_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S, h_S, s_size * sizeof(int), cudaMemcpyHostToDevice));

    // Define kernel parameters
    const int THREAD_TILE = 1;
    const int NUM_THREADS = 32; // Multiple of WARPSIZE
    const int num_blocks = (r_size + s_size + THREAD_TILE * NUM_THREADS - 1) / (THREAD_TILE * NUM_THREADS);

    // Launch kernel
    merge_path<int, THREAD_TILE, NUM_THREADS><<<num_blocks, NUM_THREADS>>>(
        d_R, d_S, r_size, s_size, d_out, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back to host
    int h_out[r_size + s_size];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (r_size + s_size) * sizeof(int), cudaMemcpyDeviceToHost));

    // // Print output
    // for (size_t i = 0; i < r_size + s_size; ++i) {
    //     std::cout << h_out[i] << " ";
    // }
    // std::cout << std::endl;

    // // Print expected output
    // for (size_t i = 0; i < r_size + s_size; ++i) {
    //     std::cout << h_comp[i] << " ";
    // }
    // std::cout << std::endl;

    for (int idx = 0; idx < r_size + s_size; idx++) {
        assert(h_out[idx] == h_comp[idx]);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_out));
}

void performance_test_3(){
    // Initialize input arrays
    constexpr int size = 1 << 18;

    const size_t r_size = size;
    const size_t s_size = size;

    int h_R[r_size];
    int h_S[s_size];

    int h_comp[r_size + s_size];

    for (int idx = 0; idx < r_size; idx++) {
        h_R[idx] = 2 * idx;
    }
    for (int idx = 0; idx < s_size; idx++) {
        h_S[idx] = 2 * idx + 1;
    }

    // Sort the arrays
    std::sort(h_R, h_R + r_size);
    std::sort(h_S, h_S + s_size);

    // Merge the arrays on the host
    int i = 0, j = 0, k = 0;
    while (i < r_size && j < s_size) {
        if (h_R[i] < h_S[j]) {
            h_comp[k++] = h_R[i++];
        } else {
            h_comp[k++] = h_S[j++];
        }
    }
    while (i < r_size) {
        h_comp[k++] = h_R[i++];
    }
    while (j < s_size) {
        h_comp[k++] = h_S[j++];
    }

    // // Print input arrays
    // std::cout << "h_R: ";
    // for (size_t i = 0; i < r_size; ++i) {
    //     std::cout << h_R[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "h_S: ";
    // for (size_t i = 0; i < s_size; ++i) {
    //     std::cout << h_S[i] << " ";
    // }
    // std::cout << std::endl;

    // Allocate device memory
    int *d_R, *d_S, *d_out;
    CUDA_CHECK(cudaMalloc(&d_R, r_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_S, s_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, (r_size + s_size) * sizeof(int)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_R, h_R, r_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S, h_S, s_size * sizeof(int), cudaMemcpyHostToDevice));

    // Define kernel parameters
    const int THREAD_TILE = 128;
    const int NUM_THREADS = 32; // Multiple of WARPSIZE
    const int num_blocks = (r_size + s_size + 2 * THREAD_TILE * NUM_THREADS - 1) / (2 * THREAD_TILE * NUM_THREADS);

    // Calculate the maximum number of resident blocks
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    int maxBlocksPerSM;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM, merge_path<int, THREAD_TILE, NUM_THREADS>, NUM_THREADS, 0));

    int maxResidentBlocks = maxBlocksPerSM * deviceProp.multiProcessorCount;
    std::cout << "Max number of resident blocks: " << maxResidentBlocks << std::endl;

    assert(num_blocks <= maxResidentBlocks);

    // Launch kernel
    // Start timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    merge_path<int, THREAD_TILE, NUM_THREADS><<<num_blocks, NUM_THREADS>>>(
        d_R, d_S, r_size, s_size, d_out, nullptr);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Clean up timing events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back to host
    int h_out[r_size + s_size];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (r_size + s_size) * sizeof(int), cudaMemcpyDeviceToHost));

    // // Print output
    // for (size_t i = 0; i < r_size + s_size; ++i) {
    //     std::cout << h_out[i] << " ";
    // }
    // std::cout << std::endl;

    // // Print expected output
    // for (size_t i = 0; i < r_size + s_size; ++i) {
    //     std::cout << h_comp[i] << " ";
    // }
    // std::cout << std::endl;

    for (int idx = 0; idx < r_size + s_size; idx++) {
        // assert(h_out[idx] == h_comp[idx]);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_out));
}

int main(){
    // visual_test_1();
    // correctness_test_2();
    performance_test_3();
}