#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <cstdint>

// Hash table implementation
template <typename KeyT, typename ValueT>
class HashTable {

    using IdxT = uint64_t;
    using ErrT = int;

    KeyT *key_array;
    ValueT *value_array;
    int size;

public:
    IdxT NOT_FOUND = (IdxT) -1;
    KeyT EMPTY = 0;
    ValueT DEFAULT = 0;

    ErrT SUCCESS = 0;
    ErrT FAILURE = 1;

    __device__ HashTable() {
        size = 0;
        key_array = nullptr;
        value_array = nullptr;
    }

    __device__ HashTable(int size) {
        this->size = size;
        key_array = new KeyT[size];
        value_array = new ValueT[size];
    }

    __device__ ~HashTable() {
        delete[] key_array;
        delete[] value_array;
    }

    size_t memory_usage() {
        return sizeof(KeyT) * size + sizeof(ValueT) * size + 1;
    }

    __device__ IdxT hash(KeyT key) {
        IdxT x = (IdxT) key;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x;
    }

    // Hash function + linear probing
    __device__ IdxT next_probe(KeyT key, int probe_idx) {
        assert(probe_idx < size);
        return (probe_idx + hash(key)) % size;
    }

    __device__ ErrT insert(KeyT key, ValueT value) {
        for (int probe_idx = 0; probe_idx < size; probe_idx++) {
            IdxT probe = next_probe(key, probe_idx);
            KeyT cur_key = atomicCAS(&key_array[probe], EMPTY, key);
            if (cur_key == EMPTY || cur_key == key) {
                value_array[probe] = value;
                return SUCCESS;
            }
        }
        return FAILURE;
    }

    // Assumes no writes are happening concurrently
    __device__ IdxT find(KeyT key) {
        for (int probe = 0; probe < size; probe++) {
            IdxT probe_idx = next_probe(key, probe);
            if (key_array[probe_idx] == key) {
                return probe_idx;
            }
        }
        return NOT_FOUND;
    }

    // Assumes no writes are happening concurrently
    __device__ ValueT get(KeyT key) {
        IdxT idx = find(key);
        if (idx == NOT_FOUND) {
            return DEFAULT;
        }
        return value_array[idx];
    }
};;

__global__ void hash_test(){
    assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.y == 1 && blockDim.z == 1);

    extern __shared__ char ht_memory[];
    HashTable<int, int>* ht = reinterpret_cast<HashTable<int, int>*>(ht_memory);
    if (threadIdx.x == 0) {
        ht[0] = HashTable<int, int>(32);
    }
    __syncthreads();
    ht[0].insert(threadIdx.x, threadIdx.x);
    __syncthreads();
    bool found = ht[0].find(threadIdx.x) != HashTable<int, int>().NOT_FOUND;
    assert(found);
    assert(ht[0].get(threadIdx.x) == threadIdx.x);
}


int main(){
    dim3 grid(1, 1, 1);
    dim3 block(32, 1, 1);
    hash_test<<<grid, block>>>();
}
