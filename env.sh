#!/bin/bash

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10


export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_VISIBLE_DEVICES="0"

echo "✅ Conda CUDA 환경 설정 완료. CUDA_HOME=$CUDA_HOME"