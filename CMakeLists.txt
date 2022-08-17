cmake_minimum_required(VERSION 3.16)
project(YOLORunner CXX C)

find_package(PkgConfig REQUIRED)

list(APPEND CMAKE_PREFIX_PATH
	dist/ort
	dist/torch
	dist/torchvision
)

pkg_check_modules(onnxruntime REQUIRED libonnxruntime)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

list(APPEND CMAKE_MODULE_PATH cmake)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")

find_package(OpenCV REQUIRED)

include(argparse)
include(ctre)
include(fmt)
include(json)
include(mio)
include(spdlog)

include(common)

include_directories(include)

add_subdirectory(examples)

add_subdirectory(private)
