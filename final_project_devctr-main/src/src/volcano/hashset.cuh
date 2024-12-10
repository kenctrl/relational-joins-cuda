#ifndef HASHTABLE_CUH
#define HASHTABLE_CUH

#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <cstdint>

// Hash table implementation
template <typename KeyT, int size>
class HashSet {

    using IdxT = uint64_t;
    using ErrT = int;

    KeyT key_array[size];

public:
    IdxT NOT_FOUND = (IdxT) 0;
    IdxT FOUND = (IdxT) 1;
    KeyT EMPTY = 0;

    ErrT SUCCESS = 0;
    ErrT FAILURE = 1;

    size_t memory_usage() {
        return sizeof(KeyT) * size + 1;
    }

    __device__ __forceinline__ IdxT hash(KeyT key) {
        IdxT x = (IdxT) key;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x;
    }

    __device__ __forceinline__ IdxT next_probe(KeyT key, int probe_idx) {
        // assert(probe_idx < size);
        return (probe_idx + hash(key)) % size;
    }

    __device__ ErrT insert(KeyT key) {
        for (int probe_idx = 0; probe_idx < size; probe_idx++) {
            IdxT probe = next_probe(key, probe_idx);
            KeyT cur_key = atomicCAS(&key_array[probe], EMPTY, key);
            if (cur_key == EMPTY || cur_key == key) return SUCCESS;
        }
        return FAILURE;
    }

    __device__ IdxT exists(KeyT key) {
        for (int probe = 0; probe < size; probe++) {
            IdxT probe_idx = next_probe(key, probe);
            if (key_array[probe_idx] == key) {
                return FOUND
            }
        }
        return NOT_FOUND;
    }
};

#endif // HASHTABLE_CUH
