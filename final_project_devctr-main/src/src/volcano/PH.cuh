#include "utils.cuh"
#include "cassert"

template <int radix_size, int block>
struct cache_t {
    int shuffle[radix_size * block];
    int value[radix_size * block];
    int end[radix_size];
    int start[radix_size];
    int head[radix_size];
};


// first 32 bits of head is the size of the bucket
// next 32 bits of head in the position of the head
// blockDim.x == radix_size
template<int radix_size, int shift, int block, int32_t bucket_size>
__global__ void first_pass(int* __restrict__ num_buckets, 
                           unsigned long long* __restrict__ head,
                           int * __restrict__ part,
                           int * __restrict__ idx,
                           int * __restrict__ value,
                           int * __restrict__ cnt,
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
            cache.value[cur] = regs[k];
        }
    }

    __syncthreads();

    int left = cache.start[tid];
    int right = cache.end[tid];
    unsigned long long length = right - left;
    while(length > 0) {
        atomicMin(&head[tid], ((unsigned long long) bucket_size) << 32);
        unsigned long long prev = atomicAdd(&head[tid], length << 32);

        const uint32_t len = prev >> 32;
        const uint32_t start = prev & 0xFFFFFFFF;
        const uint32_t cur_bucket = start / bucket_size;

        uint32_t run_length = 0;
        if(len < bucket_size) {
            if(len + length < bucket_size) {
                run_length = length;
            } else {
                run_length = bucket_size - len;

                int next_bucket = atomicAdd(num_buckets, 1);

                unsigned long long new_start = next_bucket * bucket_size;
                atomicExch(&head[tid], new_start);

                part[next_bucket] = tid;
            }
        }
        atomicAdd(&cnt[cur_bucket], run_length);

        for(int x = 0; x < run_length; x++) {
            idx[start + len + x] = cache.shuffle[left + x];
            value[start + len + x] = cache.value[left + x];
        }
        left += run_length;
        length -= run_length;
    }
}

// first 32 bits of head is the size of the bucket
// next 32 bits of head in the position of the head
// will assume blockDim.x == radix_size2;
// and gridDim.x == num_lev1_buckets
template<int radix_size1, int radix_size2, int shift, int bucket_size1, int bucket_size2>
__global__ void second_pass(
                           int *__restrict__ num_lev2_buckets,
                           unsigned long long* __restrict__ new_head,
                           int * __restrict__ new_part,
                           int * __restrict__ new_idx,
                           int * __restrict__ new_value,
                           int * __restrict__ new_cnt,
                           const int * __restrict__ part,
                           const int * __restrict__ idx,
                           const int * __restrict__ value,
                           const int * __restrict__ cnt,
                           const int *__restrict__ num_lev1_buckets) {

    constexpr int block = bucket_size1 / radix_size2;
    __shared__ cache_t<radix_size2, block> cache;
    const int bucket_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int start = bucket_size1 * bucket_id;
    const int partition = (bucket_id < *num_lev1_buckets) ? part[bucket_id] : -1;
    const int num_items = (bucket_id < *num_lev1_buckets) ? cnt[bucket_id] : 0;
    int regs[block];
    cache.end[tid] = 0;
    __syncthreads();

    #pragma unroll
    for(int k = 0; k < block; k++) {
        const int offset = k * blockDim.x + tid;
        const int i = start + offset;
        if(offset < num_items) {
            regs[k] = value[i];
            int part_idx = (regs[k] >> shift) & (radix_size2 - 1);
            atomicAdd(&cache.end[part_idx], 1);
        }
    }
    __syncthreads();
    if(tid == 0) {
        int sum = 0;
        for(int i = 0; i < radix_size2; i++) {
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
        const int offset = k * blockDim.x + tid;
        const int i = start + offset;
        if(offset < num_items) {
            const int part_idx = (regs[k] >> shift) & (radix_size2 - 1);
            int cur = atomicAdd(&cache.head[part_idx], 1);
            cache.shuffle[cur] = idx[i];
            cache.value[cur] = regs[k];
        }
    }

    __syncthreads();

    int left = cache.start[tid];
    int right = cache.end[tid];
    unsigned long long length = right - left;
    int head_idx = partition * radix_size2 + tid;

    while(length > 0) {
        atomicMin(&new_head[head_idx], ((unsigned long long) bucket_size2) << 32);
        unsigned long long prev = atomicAdd(&new_head[head_idx], length << 32);

        const uint32_t len = prev >> 32;
        const uint32_t start = prev & 0xFFFFFFFF;
        const uint32_t cur_bucket = start / bucket_size2;

        uint32_t run_length = 0;
        if(len < bucket_size2) {
            if(len + length < bucket_size2) {
                run_length = length;
            } else {
                run_length = bucket_size2 - len;

                int next_bucket = atomicAdd(num_lev2_buckets, 1);

                unsigned long long new_start = next_bucket * bucket_size2;
                atomicExch(&new_head[head_idx], new_start);

                new_part[next_bucket] = head_idx;
            }
        }
        atomicAdd(&new_cnt[cur_bucket], run_length);

        for(int x = 0; x < run_length; x++) {
            new_idx[start + len + x] = cache.shuffle[left + x];
            new_value[start + len + x] = cache.value[left + x];
        }
        left += run_length;
        length -= run_length;
    }
}


template <int radix_size, int bucket_size>
__global__ void init_buckets(int *num_buckets,
                             unsigned long long *head,
                             int* part,
                             int* cnt,
                             int max_buckets) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i == 0) {
        *num_buckets = radix_size;
    }
    if(i < max_buckets) {
        part[i] = (i < radix_size) ? i : -1;
        cnt[i] = 0;
    }
    if(i < radix_size) {
        head[i] = i * bucket_size;
    }
}


static void check_ph_correctness(const int *num_buckets,
                       const int *part,
                       const int *idx,
                       const int *cnt,
                       const int *keys, // host pointer
                       int num_rows, int max_buckets, int bucket_size,
                       std::function<int(int)> radix_fn) {

    int *h_part, *h_idx, *h_cnt, *h_num_buckets;

    CHECK_CUDA_ERROR(
        cudaMallocHost(&h_num_buckets, sizeof(int))
    );

    CHECK_CUDA_ERROR(
        cudaMallocHost(&h_part, sizeof(int) * max_buckets)
    );

    CHECK_CUDA_ERROR(
        cudaMallocHost(&h_cnt, sizeof(int) * max_buckets)
    );

    CHECK_CUDA_ERROR(
        cudaMallocHost(&h_idx, sizeof(int) * max_buckets * bucket_size)
    );

    CHECK_CUDA_ERROR(
        cudaMemcpy(h_num_buckets, num_buckets, sizeof(int), cudaMemcpyDeviceToHost)
    );

    CHECK_CUDA_ERROR(
        cudaMemcpy(h_part, part, sizeof(int) * max_buckets, cudaMemcpyDeviceToHost)
    );

    CHECK_CUDA_ERROR(
        cudaMemcpy(h_cnt, cnt, sizeof(int) * max_buckets, cudaMemcpyDeviceToHost)
    );

    CHECK_CUDA_ERROR(
        cudaMemcpy(h_idx, idx, sizeof(int) * max_buckets * bucket_size, cudaMemcpyDeviceToHost)
    );

    std::cout << "All memory moved the host" << std::endl;

    std::cout << "max buckets " << max_buckets << std::endl;
    std::cout << "buckets needed " << *h_num_buckets << std::endl;

    int num_bucks = 0;
    int total_items = 0;
    for(int i = 0; i < max_buckets; i++) {
        if(h_part[i] != -1) {
            int partition = h_part[i];
            total_items += h_cnt[i];
            for(int j = 0; j < h_cnt[i]; j++) {
                int index = h_idx[i * bucket_size + j];
                int value = keys[index];
                int recalc = radix_fn(value);
                if(recalc != partition) {
                    std::cout << "bucket no " << i << std::endl;
                    std::cout << "partition " << partition << std::endl;
                    std::cout << "Incorrect at " << index << std::endl;
                    for(int k = 0; k < h_cnt[i]; k++) {
                        std::cout << h_idx[i * bucket_size + k] << " ";
                    }
                    std::cout << std::endl;
                    assert(false);
                }
            }
            ++num_bucks;
        }
    }
    std::cout << "total buckets found: " << num_bucks << std::endl;
    std::cout << "total items found: " << total_items << std::endl;
    assert(total_items == num_rows);
    assert(num_bucks == *h_num_buckets);
    std::cout << "Correctness checks passed!" << std::endl;
}


// returns the cpu pointer
template <int radix_size, int shift, int block, int bucket_size>
static auto launch_first_pass(const int* keys, int num_rows) {
    const int max_buckets = num_rows / bucket_size + radix_size + 2;

    int *d_keys;
    int *num_buckets;
    unsigned long long *head;
    int *part;
    int *value, *idx;
    int *cnt;

    allocate_mem(&d_keys, false, sizeof(int) * num_rows);
    allocate_mem(&num_buckets, false, sizeof(int));
    allocate_mem(&head, false, sizeof(unsigned long long) * radix_size);
    allocate_mem(&part, false, sizeof(int) * max_buckets);
    allocate_mem(&idx, false, sizeof(int) * max_buckets * bucket_size);
    allocate_mem(&value, false, sizeof(int) * max_buckets * bucket_size);
    allocate_mem(&cnt, false, sizeof(int) * max_buckets);

    CHECK_CUDA_ERROR(
        cudaMemcpy(d_keys, keys, sizeof(int) * num_rows, cudaMemcpyHostToDevice)
    );

    init_buckets<radix_size, bucket_size><<<max_buckets / 32 + 1, 32>>>(
        num_buckets,
        head,
        part,
        cnt,
        max_buckets
    );

    auto launch = [&] () {
        const int gridDimX = (num_rows + radix_size * block - 1) / (radix_size * block);
        first_pass<radix_size, shift, block, bucket_size><<<gridDimX, radix_size>>> (
            num_buckets,
            head,
            part,
            idx,
            value,
            cnt,
            d_keys,
            num_rows
        );
        CHECK_LAST_CUDA_ERROR();
    };
    float t = 0;
    SETUP_TIMING()
    TIME_FUNC(launch(), t)
    std::cout << "First pass timing " << t << std::endl;

    auto radix_fn = [&] (int x) {
        return (x >> shift) & (radix_size - 1);
    };

    check_ph_correctness(num_buckets, part, idx, cnt, keys, 
                      num_rows, max_buckets, bucket_size, radix_fn);

    return std::make_tuple(
        num_buckets,
        idx,
        value,
        part,
        cnt 
    );
}

static auto launch_second_pass(int *keys, int num_rows) {
    constexpr int lev1_block = 8;
    constexpr int radix_size1 = 256;
    constexpr int radix_size2 = 256;
    constexpr int shift1 = 1;
    constexpr int shift2 = 9;
    constexpr int bucket_size1 = radix_size1 * lev1_block;
    constexpr int bucket_size2 = 1024;

    const auto [num_lev1_buckets, lev1_idx, lev1_value, lev1_part, lev1_cnt] = 
                                                        launch_first_pass<radix_size1, shift1, lev1_block, bucket_size1>(keys, num_rows);


    constexpr int radix_size = radix_size1 * radix_size2;
    const int max_buckets = num_rows / bucket_size2 + radix_size + 2;

    int *num_buckets;
    unsigned long long *head;
    int *part;
    int *value, *idx;
    int *cnt;


    int *h_num_lev1_buckets;

    CHECK_CUDA_ERROR(
        cudaMallocHost(&h_num_lev1_buckets, sizeof(int))
    );
    CHECK_CUDA_ERROR(
        cudaMemcpy(h_num_lev1_buckets, num_lev1_buckets, sizeof(int), cudaMemcpyDeviceToHost)
    );

    allocate_mem(&num_buckets, false, sizeof(int));
    allocate_mem(&head, false, sizeof(unsigned long long) * radix_size);
    allocate_mem(&part, false, sizeof(int) * max_buckets);
    allocate_mem(&idx, false, sizeof(int) * max_buckets * bucket_size2);
    allocate_mem(&value, false, sizeof(int) * max_buckets * bucket_size2);
    allocate_mem(&cnt, false, sizeof(int) * max_buckets);

    init_buckets<radix_size, bucket_size2><<<max_buckets / 32 + 1, 32>>>(
        num_buckets,
        head,
        part,
        cnt,
        max_buckets
    );
    printf("hello everyone\n");

    auto launch = [&] () {
        const int gridDimX = *h_num_lev1_buckets;
        second_pass<radix_size1, radix_size2, shift2, bucket_size1, bucket_size2><<<gridDimX, radix_size2>>> (
            num_buckets,
            head,
            part,
            idx,
            value,
            cnt,
            lev1_part,
            lev1_idx,
            lev1_value,
            lev1_cnt,
            num_lev1_buckets
        );
        CHECK_LAST_CUDA_ERROR();
    };

    float t = 0;
    SETUP_TIMING()
    TIME_FUNC(launch(), t)
    std::cout << "Second pass timing " << t << std::endl;

    auto radix_fn = [&] (int x) {
        int partition1 = (x >> shift1) & (radix_size1 - 1);
        int partition2 = (x >> shift2) & (radix_size2 - 1);
        return partition1 * radix_size2 + partition2;
    };

    check_ph_correctness(num_buckets, part, idx, cnt, keys, 
                      num_rows, max_buckets, bucket_size2, radix_fn);

    return std::make_tuple(
        num_buckets, 
        idx,
        value,
        part,
        cnt
    );
}