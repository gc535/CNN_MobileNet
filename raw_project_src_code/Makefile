# Makefile
#==========================================================================
# @brief: A makefile the compiles and synthesizes the digitrec program
#
# @desc: 1. "make" runs csim by default
#        2. "make csim" compiles & executes the fixed-point implementation
#        3. "make clean" cleans up the directory


# Extract Vivado HLS include path
VHLS_PATH := $(dir $(shell which vivado_hls))/..
VHLS_INC ?= ${VHLS_PATH}/include

INC_PATH=/usr/include/vivado_hls/2015.2
ZFLAGS = -I${INC_PATH}

CFLAGS = -g -I${VHLS_INC}

main_v3: main.cpp logit_layer.cpp batchnorm_relu.cpp loadbmp.cpp perform_conv2d.cpp 
	g++ ${CFLAGS} $^ -o main_v3 -lrt
	