#!/usr/bin/bash

# --machine, -m {32|64}: specify machine bit arch
# --generate-code -gencode <specification> : generate PTX code for virtual architectures
#     --> this is important! Need to generate code for Maxwell support
#         --> arch = compute_52, compute_53
#   --> this is a generalization of --gpu-architecture --gpu-code, see docs
# --compiler-binder -ccbin: specify path to host compiler binary

NVCC=/usr/local/cuda-10.2/bin/nvcc
NAME=vecAdd

$NVCC --compiler-bindir g++ -m64 -I/usr/local/cuda-10.2/samples/common/inc -gencode arch=compute_53,code=sm_53 -o "$NAME".o -c "$NAME".c 
$NVCC --compiler-bindir g++ -m64 -gencode arch=compute_53,code=sm_53 -o "$NAME" "$NAME".o
