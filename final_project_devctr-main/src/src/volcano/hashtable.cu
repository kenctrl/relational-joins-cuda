#include "hashtable.cuh"

__global__ void hash_test(){
    using HashTypeT = HashTable<int, int, 32>;
    assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.y == 1 && blockDim.z == 1);

    extern __shared__ char ht_memory[];
    HashTypeT* ht = reinterpret_cast<HashTypeT*>(ht_memory);
    if (threadIdx.x == 0) {
        ht[0] = HashTypeT();
    }
    __syncthreads();
    ht[0].insert(threadIdx.x, threadIdx.x);
    __syncthreads();
    bool found = ht[0].find(threadIdx.x) != ht[0].NOT_FOUND;
    assert(found);
    assert(ht[0].get(threadIdx.x) == threadIdx.x);
}


int main(){
    dim3 grid(1, 1, 1);
    dim3 block(32, 1, 1);
    hash_test<<<grid, block>>>();
}
