#include <cuda_runtime.h>

using IdxT = int;
using ErrT = int;
// Hash table implementation
template <typename KeyT, typename ValueT>
class HashTable {

    IdxT NOT_FOUND = -1;
    KeyT EMPTY = 0;
    ValueT DEFAULT = 0;

    ErrT SUCCESS = 0;
    ErrT FAILURE = 1;

public:
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

    // Hash function + linear probing
    __device__ IdxT next_probe(KeyT key, int probe_idx) {
        assert(probe_idx < size);
        return (probe_idx + 1) % size;
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
            IdxT probe_idx = (hash(key) + probe) % size;
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

__global__ int hash_test(){
    assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.y == 1 && blockDim.z == 1);

    __shared__ HashTable<int, int> ht[1];

    if (threadIdx.x == 0) {
        ht[0] = HashTable<int, int>(32);
    }
    __syncthreads();
    ht[0].insert(threadIdx.x, threadIdx.x);
    __syncthreads();
    return 0;
}


int main(){
    dim3 grid(1, 1, 1);
    dim3 block(32, 1, 1);
    hash_test<<<grid, block>>>();
}