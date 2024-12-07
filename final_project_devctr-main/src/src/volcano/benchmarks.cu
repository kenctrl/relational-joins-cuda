#include "cuda_runtime.h"
#include "utils.cuh"

#define NUM_WARPS 32
#define ALIGN(num_rows) (((num_rows) + NUM_WARPS - 1) / NUM_WARPS)
#define DEALIGN(num_cols) ((num_cols) + ((num_cols) & 1)^1)

// Assumes num_rows is divisible by blockDim.x
__global__ void copy(int* row_major, const int * __restrict__ col_major, int num_rows, int num_cols) {
    extern __shared__ int cache[];
    const int start_row = blockIdx.x * blockDim.x;
    const int tid = threadIdx.x;

    for(int j = 0; j < num_cols; j++) {
        cache[tid * num_cols + j] = col_major[j * num_rows + start_row + tid];  
    }
    __syncthreads();
    for(int k = tid; k < blockDim.x * num_cols; k += blockDim.x) {
        const int row = k / num_cols;
        const int col = k % num_cols;
        row_major[start_row * num_cols + k] = cache[row * num_cols + col];
    } 
}

int* transform_layout(int *col_major, int num_rows, int num_cols) {
    int *d_col_major, *d_row_major;
    allocate_mem(&d_col_major, false, sizeof(int) * num_rows * num_cols);
    allocate_mem(&d_row_major, false, sizeof(int) * num_rows * num_cols);
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_col_major, col_major, sizeof(int) * num_rows * num_cols, cudaMemcpyHostToDevice)
    );

    auto launch = [&] () {
        copy<<<num_rows / NUM_WARPS, NUM_WARPS, sizeof(int) * NUM_WARPS * num_cols>>>(
            d_row_major,
            d_col_major,
            num_rows,
            num_cols
        );
    };

    float t = 0;
    SETUP_TIMING()
    TIME_FUNC(launch(), t)

    printf("Time taken: %f\n", t);

    release_mem(d_col_major);
    return d_row_major;
}

int main() {
    const int num_rows = 1 << 25;
    const int num_cols = 4;
    int *col_major;
    CHECK_CUDA_ERROR(
        cudaMallocHost(&col_major, sizeof(int) * num_rows * num_cols)
    );

    for(int j = 0; j < num_cols; j++) {
        for(int i = 0; i < num_rows; i++) {
            col_major[j * num_rows + i] = j * num_rows + i;
        }
    }

    int* d_row_major = transform_layout(col_major, num_rows, num_cols);
    int *row_major;

    CHECK_CUDA_ERROR(
        cudaMallocHost(&row_major, sizeof(int) * num_rows * num_cols)
    );

    CHECK_CUDA_ERROR(
        cudaMemcpy(row_major, d_row_major, sizeof(int) * num_rows * num_cols, cudaMemcpyDeviceToHost)
    );

    for(int i = 0; i < num_rows && i < 64; i++) {
        for(int j = 0; j < num_cols; j++) {
            printf("%d ", row_major[i * num_cols + j]);
        }
        printf("\n");
    }

    CHECK_CUDA_ERROR(
        cudaFreeHost(col_major)
    );
}