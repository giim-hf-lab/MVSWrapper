#!/usr/bin/env bash
set -eu

installed_path="build/_installed"

cmake --toolchain ${PWD}/toolchains/ubuntu/gcc-11.cmake -G Ninja -B build/project .

cmake --build build/project --config Release --target $*
