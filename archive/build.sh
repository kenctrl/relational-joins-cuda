#!/bin/bash

# Output build directory path.
CTR_BUILD_DIR=/build

echo "Building the project..."

# ----------------------------------------------------------------------------------
# ------------------------ PUT YOUR BULDING COMMAND(s) HERE ------------------------
# ----------------------------------------------------------------------------------
# ----- This sctipt is executed inside the development container:
# -----     * the current workdir contains all files from your src/
# -----     * all output files (e.g. generated binaries, test inputs, etc.) must be places into $CTR_BUILD_DIR
# ----------------------------------------------------------------------------------
# Build code.
NVCCFLAGS += --std=c++17 --expt-relaxed-constexpr --expt-extended-lambda --extended-lambda $(SM_DEF) -Xptxas="-v" -lineinfo -Xcudafe -\# 

SRC=src/src
BIN=src/bin
BUILD_DIR=src/obj
CU_SRC:=$(shell find $(SRC) -name "*.cu")
OBJ:=$(CU_SRC:src/%.cu=$(BUILD_DIR)/%.o)
DEP:=$(OBJ:%.o=%.d)

CUB_DIR=src/cub/

INCLUDES =-I$(CUB_DIR) -I.

$(BIN)/%: $(BUILD_DIR)/%.o
	$(NVCC) $(SM_TARGETS) -Xlinker -lgomp $< -o $@

-include $(DEP)

$(BUILD_DIR)/%.o: $(SRC)/%.cu
	$(NVCC) -Xcompiler -fopenmp -MMD $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 --compile $< -o $@

nvcc -O3 -arch=sm_60 vector_add.cu -o ${CTR_BUILD_DIR}/vector_add
