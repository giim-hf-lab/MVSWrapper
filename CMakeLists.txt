cmake_minimum_required(VERSION 3.16)
project(YOLORunner CXX C)

find_package(OpenCV REQUIRED)

list(APPEND CMAKE_PREFIX_PATH
	"${CMAKE_CURRENT_SOURCE_DIR}/dist/torch"
	"${CMAKE_CURRENT_SOURCE_DIR}/dist/torchvision"
)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")

include(onnxruntime)

include(argparse)
include(fmt)
include(mio)
include(spdlog)

include(common)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_libraries(
	argparse::argparse
	fmt::fmt
	mio::mio
	spdlog::spdlog
)

add_executable(images_torch src/images/torch.cpp)
target_include_directories(images_torch PRIVATE
	${TORCH_INCLUDE_DIRS}
	${TorchVision_INCLUDE_DIR}
)
target_link_libraries(images_torch PRIVATE
	opencv_core
	opencv_highgui
	opencv_imgcodecs
	opencv_imgproc

	${TORCH_LIBRARIES}
	TorchVision::TorchVision
)

add_executable(images_ort src/images/ort.cpp)
target_include_directories(images_ort PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/externals/onnxruntime/include"
)
add_dependencies(images_ort onnxruntime_providers_cuda)
target_link_libraries(images_ort PRIVATE
	opencv_core
	opencv_highgui
	opencv_imgcodecs
	opencv_imgproc

	onnxruntime
)
