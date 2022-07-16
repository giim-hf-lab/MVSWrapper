#!/usr/bin/env bash
set -eu

installed_path="build/_installed"

cmake --toolchain toolchains/gnu/host.cmake -G Ninja -B build/project .

cmake --build build/project --target detect
