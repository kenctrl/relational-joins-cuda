#include "iostream"
#include "tuple"
#include "type_traits"
#include "array"
#include <cuda_runtime.h>
#include <stdio.h>
#include <numeric>
#include "utils.cuh"

template <std::size_t... Indices, typename Func>
constexpr void index_folding(std::index_sequence<Indices...>, Func&& func) {
    (func(std::integral_constant<std::size_t, Indices>{}), ...);
}

template <std::size_t Start, std::size_t End, typename Func>
constexpr void static_for(Func&& func) {
    index_folding(std::make_index_sequence<End - Start>{}, std::forward<Func>(func));
}

template <typename...Args> 
__global__ void row_major_order(int num_rows, char* data, Args*... args) {
    constexpr int row_size = (sizeof(Args) + ... + 0);
    const auto columns = std::make_tuple(args...);
    using ColumnPointers = std::tuple<Args*...>;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += blockDim.x) {
        int offset = 0;
        static_for<0, sizeof...(Args)>([&] (auto item) {
            constexpr int index = item;
            using PointerType = typename std::tuple_element<index, ColumnPointers>::type;
            using ValueType = typename std::remove_pointer<PointerType>::type; 
            auto col = std::get<index>(columns);
            *(PointerType)(data + i * row_size + offset) = col[i];
            offset += sizeof(ValueType);
        });
    }
}

template <typename...Args> 
__global__ void row_printer(int num_rows, char* data, Args*... args) {
    constexpr int row_size = (sizeof(Args) + ... + 0);
    const auto columns = std::make_tuple(args...);
    using ColumnPointers = std::tuple<Args*...>;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += blockDim.x) {
        int offset = 0;
        static_for<0, sizeof...(Args)>([&] (auto item) {
            constexpr int index = item;
            using PointerType = typename std::tuple_element<index, ColumnPointers>::type;
            using ValueType = typename std::remove_pointer<PointerType>::type; 
            auto col = std::get<index>(columns);
            if constexpr(index == 2) {
                printf("%d\n", *(PointerType)(data + i * row_size + offset));
            }
            offset += sizeof(ValueType);
        });
    }
}

template <typename... Args>
class Table {
public:
    using ColumnPointers = std::tuple<Args*...>;

    int num_rows;
    std::tuple<Args*...> device_cols;

    static constexpr int row_size = (sizeof(Args) + ... + 0);
    static constexpr int num_cols = sizeof...(Args);

    Table(int num_rows, cudaStream_t stream, Args*... args) : num_rows(num_rows) {
        ColumnPointers host_cols = std::make_tuple(args...);
        send_to_gpu(host_cols, stream);
        create_row_major(stream);
    }

    void send_to_gpu(ColumnPointers& host_cols, cudaStream_t stream) {
        static_for<0, num_cols>([&] (auto i) {
            constexpr int index = i;
            using PointerType = typename std::tuple_element<index, ColumnPointers>::type;
            using ValueType = typename std::remove_pointer<PointerType>::type; 

            PointerType host_ptr = std::get<index>(host_cols);
            PointerType &device_ptr = std::get<index>(device_cols);

            allocate_mem(&device_ptr, false, sizeof(ValueType) * num_rows, 0);

            CHECK_CUDA_ERROR(
                cudaMemcpyAsync(device_ptr, host_ptr, sizeof(ValueType) * num_rows, cudaMemcpyHostToDevice, stream)
            );
        });
    }

    void create_row_major(cudaStream_t stream) {
        char* data;
        allocate_mem(&data, false, row_size * num_rows, 0);
        std::apply([&] (auto... args) {
            row_major_order<<<1, 32, 0, stream>>>(num_rows, data, args...);
        }, device_cols);
        CHECK_CUDA_ERROR(cudaGetLastError());

        std::apply([&] (auto... args) {
            row_printer<<<1, 32, 0, stream>>>(num_rows, data, args...);
        }, device_cols);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    void free() {
        static_for<0, num_cols>([&] (auto item) {
            constexpr int index = item;
            release_mem(std::get<index>(device_cols), 0);
        });
    } 

    template<int N>
    auto get_device_pointer() {
        return std::get<N>(device_cols);
    }
};

template <typename... Args>
class TableStream {
public:
    using ColumnPointers = std::tuple<Args*...>;
    static constexpr int num_cols = sizeof...(Args);

    TableStream(int num_rows, Args*... args) {
        auto host_cols = std::make_tuple(args...);
        int block = (num_rows + num_cols - 1) / num_cols;

        int buf = 0;
        cudaStream_t stream[2];
        cudaStreamCreate(&stream[1]);
        CHECK_LAST_CUDA_ERROR();

        for(int i = 0; i < num_rows; i += block) {
            auto start = std::make_tuple(args...);
            static_for<0, num_cols>([&] (auto item) {
                constexpr int index = item;
                using PointerType = typename std::tuple_element<index, ColumnPointers>::type;
                using ValueType = typename std::remove_pointer<PointerType>::type; 
                PointerType &ptr = std::get<index>(start);
                ptr = ptr + i;
            });
            cudaStreamCreate(&stream[buf]);
            CHECK_LAST_CUDA_ERROR();

            auto table_size = i + block <= num_rows ? block : num_rows - i;
            auto new_args = std::tuple_cat(std::make_tuple(table_size, stream[buf]), start);
            auto table = std::make_from_tuple<Table<Args...>>(new_args);

            buf ^= 1;
            cudaStreamSynchronize(stream[buf]);
            CHECK_LAST_CUDA_ERROR();

            table.free();

            cudaStreamDestroy(stream[buf]);
            CHECK_LAST_CUDA_ERROR();
        }
        cudaStreamSynchronize(stream[buf ^ 1]);
        CHECK_LAST_CUDA_ERROR();
        cudaStreamDestroy(stream[buf ^ 1]);
        CHECK_LAST_CUDA_ERROR();
    }
};

int main() {
    constexpr int num_cols = 5;
    int num_rows = 1000;
    int *col[num_cols];
    for(int j = 0; j < num_cols; j++) {
        CHECK_CUDA_ERROR(
            cudaMallocHost(&col[j], sizeof(*col[0]) * num_rows)
        );
    }

    for(int j = 0; j < num_cols; j++) {
        for(int i = 0; i < num_rows; i++) {
            col[j][i] = i + j * num_rows;
        }
    }

    TableStream<int, int, int, int, int> table (num_rows, col[0], col[1], col[2], col[3], col[4]);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
}

