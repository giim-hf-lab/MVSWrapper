#!/usr/bin/env bash
set -eu

install_prefix="${2}"
generator="${3:-Ninja}"

mkdir -p "${install_prefix}"

ort_path="$(realpath ${install_prefix}/ort)"

cmake --toolchain "${PWD}/toolchains/${1}/gcc-11.cmake" -G "${generator}" -B build/ort -Wno-dev \
	-DCMAKE_C_STANDARD=17 \
	-DCMAKE_CXX_STANDARD=17 \
	-DCMAKE_CUDA_STANDARD=17 \
	-DCMAKE_INSTALL_PREFIX="$ort_path" \
	-Donnxruntime_BUILD_SHARED_LIB=ON \
	-Donnxruntime_USE_CUDA=ON \
	-Donnxruntime_CUDNN_HOME="/usr" \
	-Donnxruntime_USE_AVX=ON \
	-Donnxruntime_USE_AVX2=ON \
	-Donnxruntime_USE_AVX512=ON \
	-Donnxruntime_ENABLE_LTO=ON \
	-Donnxruntime_USE_NCCL=ON \
	externals/onnxruntime/cmake

cmake --build build/ort --config Release --target install

torch_path="$(realpath ${install_prefix}/torch)"

cmake --toolchain "${PWD}/toolchains/${1}/gcc-11.cmake" -G "${generator}" -B build/torch -Wno-dev \
	-DCMAKE_C_STANDARD=17 \
	-DCMAKE_CXX_STANDARD=17 \
	-DCMAKE_CUDA_STANDARD=17 \
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
	-DUSE_LITE_INTERPRETER_PROFILER=OFF \
	-DUSE_STATIC_MKL=ON \
	-DUSE_DISTRIBUTED=OFF \
	externals/torch/pytorch

cmake --build build/torch --config Release --target install

torchvision_path="$(realpath ${install_prefix}/torchvision)"

cmake --toolchain "${PWD}/toolchains/${1}/gcc-11.cmake" -G "${generator}" -B build/torchvision -Wno-dev \
	-DCMAKE_C_STANDARD=17 \
	-DCMAKE_CXX_STANDARD=17 \
	-DCMAKE_CUDA_STANDARD=17 \
	-DCMAKE_INSTALL_PREFIX="$torchvision_path" \
	-DCMAKE_PREFIX_PATH="$torch_path" \
	-DWITH_CUDA=ON \
	externals/torch/vision

cmake --build build/torchvision --config Release --target install
