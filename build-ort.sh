#!/usr/bin/env bash
set -eu

install_prefix=dist

mkdir -p "$install_prefix"

onnxruntime_path="$(realpath $install_prefix/onnxruntime)"

cmake --toolchain ${PWD}/toolchains/ubuntu.cmake -G Ninja -B build/onnxruntime \
	-DCMAKE_INSTALL_PREFIX="$onnxruntime_path" \
	-Donnxruntime_CUDNN_HOME=/usr \
	-Donnxruntime_USE_CUDA=ON \
	-Donnxruntime_USE_AVX=ON \
	-Donnxruntime_USE_AVX2=ON \
	-Donnxruntime_USE_AVX512=ON \
	-Donnxruntime_ENABLE_LTO=ON \
	-Donnxruntime_USE_NCCL=ON \
	-Donnxruntime_USE_MPI=ON \
	externals/onnxruntime/cmake

cmake --build build/onnxruntime --target install
