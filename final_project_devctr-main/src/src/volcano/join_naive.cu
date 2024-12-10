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
    __shared__ HashSetT hashset;

    // Map partitions to blocks

    for (auto iter_R = 0; iter_R < PR; iter_R++) {
        for (auto rel_R = 0; rel_R < PART_SIZE; rel_R++) {
            hashset.insert(key_r[iter_R * PART_SIZE + rel_R]);
        }
        for (auto iter_S = 0; iter_S < PS; iter_S++) {
            for (auto rel_S = 0; rel_S < PART_SIZE; rel_S++) {
                if (hashset.exists(key_s[iter_S * PART_SIZE + rel_S]) == hashset.FOUND) {
                    // Do something
                }
            }
        }
        hashset = HashSetT();
    }
}
   