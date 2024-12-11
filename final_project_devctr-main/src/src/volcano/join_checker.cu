#include <iostream>
#include <stdio.h>
#include <stdlib.h>

template <typename KeyT>
void cpu_join(
    KeyT* keys_R,
    KeyT* keys_S,
    size_t size_R,
    size_t size_S,
    std::pair<KeyT, KeyT>* result,
    int& out_size){

    int out_ctr = 0;
    for (size_t i = 0; i < size_R; i++){
        for (size_t j = 0; j < size_S; j++){
            if (keys_R[i] == keys_S[j]){
                result[out_ctr++] = std::make_pair(keys_R[i], keys_S[j]);
            }
        }
    }
    out_size = out_ctr;
}

void test(){
    int size_R = 5;
    int size_S = 5;
    int* keys_R = (int*)malloc(size_R * sizeof(int));
    int* keys_S = (int*)malloc(size_S * sizeof(int));
    std::pair<int, int>* result = (std::pair<int, int>*)malloc(size_R * size_S * sizeof(std::pair<int, int>));

    for (int i = 0; i < size_R; i++){
        keys_R[i] = i;
    }
    for (int i = 0; i < size_S; i++){
        keys_S[i] = i + 2;
    }

    int out_size;
    cpu_join(keys_R, keys_S, size_R, size_S, result, out_size);

    for (int i = 0; i < out_size; i++){
        std::cout << result[i].first << " " << result[i].second << std::endl;
    }
}

int main(){
    test();
}