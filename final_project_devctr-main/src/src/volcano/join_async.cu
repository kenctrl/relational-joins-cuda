#include <cuda_runtime.h>
#include "hashset.cuh"
#include <cassert>
#include <cooperative_groups.h>
#include <cuda/barrier>

#define WARPSIZE 32

// Given list of keys, materialize the corresponding values from the right Set
template <
    typename KeyT, 
    typename ValueT,
    typename IdxT,
    int NUM_WARPS,
    int PART_SIZE,
    int HASH_SIZE,
    int PR,
    int PS>
__global__ void materialize_async(
    KeyT *key_r, 
    KeyT *key_s, 
    IdxT *idx_r,
    IdxT *idx_s,
    IdxT *idx_r_out,
    IdxT *idx_s_out) {

    // Launch only 2 warps per block
    
    using HashSetT = HashSet<KeyT, HASH_SIZE>;
    using barrier = cuda::barrier<cuda::thread_scope_block>;

    constexpr int PRODUCER = 0;
    constexpr int CONSUMER = 1;

    constexpr int NUM_PRODUCERS = 1;
    constexpr int NUM_CONSUMERS = 1;

    // Map partitions to blocks
    // Right keys might be larger than left keys

    auto warp_id = threadIdx.x / 32;
    auto role = warp_id % 2;
    auto role_idx = warp_id / 2;

    extern __shared__ int cache[];
    HashSetT* hashsets = (HashSetT*) cache; // 2 hash Sets
    barrier* bar = (barrier*) (hashsets[2]); // For pair of producer-consumer
    KeyT* shared_keys_S = (KeyT*) (bar + 1); // For streaming join double buffer

    // HashSet barrier initialization
    if (threadIdx.x == 0) {
        cooperative_groups::init(bar, NUM_WARPS * WARPSIZE);
    }

    // Create groups from warps
    cooperative_groups::thread_group warp_group 
        = cooperative_groups::tiled_partition(
            cooperative_groups::this_thread_block(), WARPSIZE);

    // Warp specialize
    // 1. Create hashSet for each partition
    // 2. Perform streaming match

    // HashSet double buffer
    int stage = 0;
    // Consumer double buffer
    int consumer_db = 0;

    if (role == PRODUCER) {
       for (auto rel_R = 0; rel_R < PART_SIZE; rel_R += WARPSIZE * NUM_PRODUCERS) {
            hashsets[stage].insert(key_r[base_R + rel_R]);
        }
    } else if (role == CONSUMER) {
        // Spawn consumer memory copy
        // Need PART_SIZE keys total
        auto load_size = PART_SIZE / NUM_CONSUMERS;
        cooperative_groups::memcpy_async(
            shared_keys_S[consumer_db] + role_idx * load_size, 
            key_s + role_idx * load_size, 
            load_size * sizeof(KeyT),
            warp_group);
    }
    barrier::arrival_token token = bar->arrive();
    // Iterate over partitions of R
    for (auto iter_R = 0; iter_R < PR; iter_R++) {
        if (role == PRODUCER) {
            // Generate hash Set for R
            auto base_R = iter_R * PART_SIZE;
            for (auto rel_R = 0; rel_R < PART_SIZE; rel_R += WARPSIZE * NUM_PRODUCERS) {
                hashsets[stage ^ 1].insert(key_r[base_R + rel_R]);
            }
            bar->wait(std::move(token)); // Need to wait for prev to be free
        } else if (role == CONSUMER) {
            // Streaming S join
            bar->wait(std::move(token));
            for (auto iter_S = 0; iter_S < PS; iter_S++) {
                auto base_S = iter_S * PART_SIZE; // Common base for all warps
                auto load_size = PART_SIZE / NUM_CONSUMERS;
                memcpy_async(
                    shared_keys_S[consumer_db ^ 1] + role_idx * load_size, 
                    key_s + base_S + role_idx * load_size, 
                    load_size * sizeof(KeyT),
                    warp_group);
                cooperative_groups::wait_prior<1>(warp_group);
                for (auto rel_S = 0; rel_S < PART_SIZE / NUM_CONSUMERS; rel_S += WARPSIZE) {
                    auto key = shared_keys_S[consumer_db][role_idx * load_size + rel_S];
                    auto probe = hashsets[stage].exists(key);
                    if (probe != hashsets[stage].NOT_FOUND) {
                        // Do something
                    }
                }
                consumer_db ^= 1;
            }
        }
        token = bar->arrive();
        stage ^= 1;
    }
}
   