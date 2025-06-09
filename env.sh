#!/bin/bash

# ---------- CUDA 환경 ----------
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# ---------- Conda 가상환경 관련 (예: dexgraspnet) ----------
export LD_LIBRARY_PATH=$HOME/.conda/envs/dexgraspnet/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# ---------- CUDA 컴파일러 버전 충돌 회피용 (nvcc + gcc) ----------
# gcc-10로 명시 (nvcc가 gcc >10을 싫어함)
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10

# ---------- PyTorch 환경 변수 (선택사항) ----------
# PyTorch Extension 컴파일 시 ABI 버전 맞춤
export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

# ---------- CUDA 아키텍처 명시 (선택사항) ----------
export TORCH_CUDA_ARCH_LIST="8.9"  # 예: RTX 30xx 시리즈
export CUDA_VISIBLE_DEVICES="0"

echo "환경변수 설정 완료. CUDA_HOME=$CUDA_HOME"
