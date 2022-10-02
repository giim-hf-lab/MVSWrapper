#!/usr/bin/env bash
set -eu

cmake --toolchain ${PWD}/toolchains/ubuntu/gcc-11.cmake -G Ninja -B build \
	-DCMAKE_INSTALL_PREFIX=dist \
	-Dinferences_BUILD_EXAMPLES:BOOL=ON \
	-Dinferences_BUILD_INTERNAL:BOOL=ON \
	.

cmake --build build --config Release
