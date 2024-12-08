#include "utils.cuh"


template <int radix_size, int block>
struct cache_t {
    int shuffle[radix_size * block];
    int end[radix_size];
    int start[radix_size];
    int head[radix_size];
};

// assumes mask == threadIdx.x
// first 32 bits of heads is the size of the bucket
// next 32 bits of heads in the position of the head
template<int radix_size, int shift, int block, int32_t bucket_size>
__global__ void first_pass(int* num_buckets, 
                           unsigned long long* heads,
                           int *chain,
                           int *buckets,
                           const int* __restrict__ keys, 
                           int num_rows) {
    __shared__ cache_t<radix_size, block> cache;
    const int start = blockIdx.x * blockDim.x * block;
    const int tid = threadIdx.x;
    int regs[block];
    cache.end[tid] = 0;
    __syncthreads();

    #pragma unroll
    for(int k = 0; k < block; k++) {
        const int i = start + k * blockDim.x + tid;
        if(i < num_rows) {
            regs[k] = keys[i];
            int idx = (regs[k] >> shift) & (radix_size - 1);
            atomicAdd(&cache.end[idx], 1);
        }
    }
    __syncthreads();
    if(tid == 0) {
        int sum = 0;
        for(int i = 0; i < radix_size; i++) {
            int cnt = cache.end[i];
            cache.start[i] = sum;
            cache.head[i] = sum;
            sum += cnt;
            cache.end[i] = sum;
        }
    }
    __syncthreads();

    #pragma unroll
    for(int k = 0; k < block; k++) {
        const int i = start + k * blockDim.x + tid;
        if(i < num_rows) {
            const int idx = (regs[k] >> shift) & (radix_size - 1);
            int cur = atomicAdd(&cache.head[idx], 1);
            cache.shuffle[cur] = i;
        }
    }

    __syncthreads();

    int left = cache.start[tid];
    int right = cache.end[tid];
    unsigned long long length = right - left;
    while(length > 0) {
        // atomicMin(&heads[tid], ((unsigned long long) bucket_size) << 32);
        unsigned long long prev = atomicAdd(&heads[tid], length << 32);
        // printf("Old head %llu, added %llu\n", prev, length << 32);

        const uint32_t len = prev >> 32;
        const uint32_t start = prev & 0xFFFFFFFF;
        uint32_t run_length = 0;
        // printf("iteration::: try bucket block len %llu bucket len %d start %d\n", length, len, start);
        if(len < bucket_size) {
            // bucket can be filled with everything
            if(len + length < bucket_size) {
                // printf("ending length is %llu\n", len + length);
                run_length = length;
            } else {
                run_length = bucket_size - len;

                int cur_bucket = start / bucket_size;
                int next_bucket = atomicAdd(num_buckets, 1);

                unsigned long long new_start = next_bucket * bucket_size;
                atomicExch(&heads[tid], new_start);

                // printf("Created new bucket at %d, head[%d] is now %llu\n", next_bucket, tid, new_start);

                chain[cur_bucket] = next_bucket;
            }
        }
        for(int x = 0; x < run_length; x++) {
            // printf("Writing index %d at bucket %d position %d which is of partition %d\n", cache.shuffle[left + x], start / bucket_size, start + len + x, tid);
            buckets[start + len + x] = cache.shuffle[left + x];
        }
        left += run_length;
        length -= run_length;
    }
}

template <int radix_size, int bucket_size>
__global__ void init_buckets(int *num_buckets,
                             unsigned long long *heads,
                             int* chain,
                             int max_buckets) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i == 0) {
        *num_buckets = radix_size;
    }
    if(i < max_buckets) {
        chain[i] = -1;
    }
    if(i < radix_size) {
        heads[i] = i * bucket_size;
    }
}


// returns the cpu pointer
int* launch_first_pass(const int* keys, int num_rows) {
    constexpr int radix_size = 256;
    constexpr int shift = 24;
    constexpr int bucket_size = 1024;
    const int max_buckets = num_rows / bucket_size + radix_size + 1;

    int *d_keys;
    int *num_buckets;
    unsigned long long *heads;
    int *chain;
    int *buckets;

    allocate_mem(&d_keys, false, sizeof(int) * num_rows);
    allocate_mem(&num_buckets, false, sizeof(int));
    allocate_mem(&heads, false, sizeof(unsigned long long) * radix_size);
    allocate_mem(&chain, false, sizeof(int) * max_buckets);
    allocate_mem(&buckets, false, sizeof(int) * max_buckets * bucket_size);

    CHECK_CUDA_ERROR(
        cudaMemcpy(d_keys, keys, sizeof(int) * num_rows, cudaMemcpyHostToDevice)
    );

    init_buckets<radix_size, bucket_size><<<max_buckets / 32 + 1, 32>>>(
        num_buckets,
        heads,
        chain,
        max_buckets
    );
    printf("hello everyone\n");

    auto launch = [&] () {
        constexpr int block = 8;
        const int gridDimX = (num_rows + radix_size * block - 1) / (radix_size * block);
        first_pass<radix_size, shift, block, bucket_size><<<gridDimX, radix_size>>> (
            num_buckets,
            heads,
            chain,
            buckets,
            d_keys,
            num_rows
        );
        CHECK_LAST_CUDA_ERROR();
    };
    float t = 0;
    SETUP_TIMING()
    TIME_FUNC(launch(), t)
    std::cout << "First pass timing " << t << std::endl;
    exit(0);
    return nullptr;
}

int main() {
    const int num_rows = 1 << 25;
    int *keys;
    CHECK_CUDA_ERROR(
        cudaMallocHost(&keys, sizeof(int) * num_rows)
    );

    int cpu[256] = { 0 };
    for(int i = 0; i < num_rows; i++) {
        keys[i] = i;
        cpu[(keys[i] >> 24) & 255] += 1;
    }

    int *cnt = launch_first_pass(keys, num_rows);

    for(int i = 0; i <= 0xFF; i++) {
        std::cout << cnt[i] << " ";
    }
    std::cout << std::endl;

    for(int i = 0; i <= 0xFF; i++) {
        std::cout << cpu[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}