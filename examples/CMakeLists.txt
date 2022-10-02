cmake_minimum_required(VERSION 3.14)
cmake_policy(VERSION 3.14)
project(examples LANGUAGES CXX VERSION ${MASTER_PROJECT_VERSION})

find_package(inferences REQUIRED)

find_package(OpenCV REQUIRED)

find_package(argparse REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

find_package(OpenCV REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(common)

add_executable(yolo yolo.cpp)
target_link_libraries(yolo PRIVATE
	argparse::argparse
	fmt::fmt
	opencv_core
	opencv_highgui
	opencv_imgcodecs
	opencv_videoio
	spdlog::spdlog

	inferences::framework::torchscript
)

add_executable(ocr ocr.cpp)
target_include_directories(ocr PRIVATE
	${onnxruntime_PREFIX}/include
)
target_link_libraries(ocr PRIVATE
	argparse::argparse
	fmt::fmt
	opencv_core
	opencv_highgui
	opencv_imgcodecs
	opencv_videoio
	spdlog::spdlog

	inferences::framework::onnxruntime
)

install(
	TARGETS
		ocr
		yolo
)
