cmake_minimum_required(VERSION 3.16)
project(YOLORunner CXX C)

list(APPEND CMAKE_PREFIX_PATH
	"${CMAKE_CURRENT_SOURCE_DIR}/dist/torch"
	"${CMAKE_CURRENT_SOURCE_DIR}/dist/torchvision"
)

find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")

include(argparse)
include(fmt)
include(mio)
include(spdlog)

include(common)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_libraries(
	argparse::argparse
	fmt::fmt
	spdlog::spdlog
)

add_executable(images src/images.cpp)
target_include_directories(images PRIVATE
	${TORCH_INCLUDE_DIRS}
	${TorchVision_INCLUDE_DIR}
)
target_link_libraries(images PRIVATE
	mio::mio
	opencv_core
	opencv_highgui
	opencv_imgcodecs
	opencv_imgproc

	${TORCH_LIBRARIES}
	TorchVision::TorchVision
)
