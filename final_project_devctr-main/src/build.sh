#!/bin/bash

# Output build directory path.
CTR_BUILD_DIR=/build

echo "Building the project..."

# Variables from Makefile
CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
CUDA_INC_PATH=${CUDA_INC_PATH:-${CUDA_PATH}/include}
CUDA_BIN_PATH=${CUDA_BIN_PATH:-${CUDA_PATH}/bin}

NVCC=nvcc

# For RTX 3090
SM_TARGETS="-gencode=arch=compute_86,code=\"sm_86,compute_86\""
SM_DEF="-DSM860"

# For A100 (uncomment if needed)
# SM_TARGETS="-gencode=arch=compute_80,code=\"sm_80,compute_80\""
# SM_DEF="-DSM800"

NVCCFLAGS="--std=c++17 --expt-relaxed-constexpr --expt-extended-lambda --extended-lambda ${SM_DEF} -Xptxas=\"-v\" -lineinfo -Xcudafe -#"

# Directories
BUILD_DIR=${CTR_BUILD_DIR}/obj
BIN_DIR=${CTR_BUILD_DIR}/bin
CUB_DIR=cub/

INCLUDES="-I${CUB_DIR} -I."

# Create necessary directories
mkdir -p ${BIN_DIR} ${BUILD_DIR}

# Find all .cu files in current directory and subdirectories
CU_SRC=$(find . -name "*.cu")

# For each .cu file, compile to .o and then link to executable
for cu_file in ${CU_SRC}; do
    # Get the relative path and base name
    rel_path=${cu_file#./}
    base_name=${rel_path%.cu}
    
    # Object file and output executable paths
    base_name=${base_name#src/}
    obj_file=${BUILD_DIR}/${base_name}.o
    output_exe=${BIN_DIR}/${base_name}

    # Create directories for object file and executable
    mkdir -p $(dirname ${obj_file})
    mkdir -p $(dirname ${output_exe})

    # If executable already exists in ./cache/bin/, then copy it over and skip compilation
    # cache_dir=./cache/bin/
    # # TODO: add manual list of files to rebuild
    # if [ -f ${cache_dir}${base_name} ]; then
    #     echo "Copying ${cache_dir}${base_name} to ${output_exe}"
    #     cp ${cache_dir}${base_name} ${output_exe}
    #     continue
    # fi

    # Compile .cu file to object file
    echo "Compiling ${cu_file} to ${obj_file}"
    ${NVCC} -Xcompiler -fopenmp -MMD ${SM_TARGETS} ${NVCCFLAGS} ${INCLUDES} -O3 --compile ${cu_file} -o ${obj_file}

    # Link object file to create executable
    echo "Linking ${obj_file} to create ${output_exe}"
    ${NVCC} ${SM_TARGETS} -Xlinker -lgomp ${obj_file} -o ${output_exe}
done

# Copy the exp/ directory
cp -r exp/ ${CTR_BUILD_DIR}/exp

# Copy the requirements.txt
cp requirements.txt ${CTR_BUILD_DIR}/requirements.txt

# Delete object files
rm -rf ${BUILD_DIR}