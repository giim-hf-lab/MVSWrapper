#!/usr/bin/env bash
set -eu

install_prefix=dist

mkdir -p "$install_prefix"

torch_path="$(realpath $install_prefix/torch)"

cmake --toolchain ${PWD}/toolchains/gnu/host.cmake -G Ninja -B build/torch \
	-DCMAKE_INSTALL_PREFIX="$torch_path" \
	-DATEN_NO_TEST=ON \
	-DBUILD_PYTHON=OFF \
	-DBUILD_CAFFE2=ON \
	-DBUILD_CAFFE2_OPS=ON \
	-DBUILD_LAZY_CUDA_LINALG=OFF \
	-DBUILD_NVFUSER_BENCHMARK=OFF \
	-DUSE_KINETO=OFF \
	-DUSE_NVRTC=ON \
	-DUSE_C10D_NCCL=ON \
	-DUSE_NCCL_WITH_UCC=ON \
	-DUSE_C10D_MPI=ON \
	externals/torch/pytorch

cmake --build build/torch --target install

torchvision_path="$(realpath $install_prefix/torchvision)"

cmake --toolchain ${PWD}/toolchains/gnu/host.cmake -G Ninja -B build/torchvision \
	-DCMAKE_INSTALL_PREFIX="$torchvision_path" \
	-DCMAKE_PREFIX_PATH="$torch_path" \
	-DWITH_CUDA=ON \
	externals/torch/vision

cmake --build build/torchvision --target install
