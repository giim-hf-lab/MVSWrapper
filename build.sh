#!/usr/bin/env bash
set -eu

installed_path="build/_installed"

cmake --toolchain ${PWD}/toolchains/ubuntu.cmake -G Ninja -B build/project .

cmake --build build/project --target $*
