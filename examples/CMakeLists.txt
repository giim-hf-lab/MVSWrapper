add_executable(yolo yolo.cpp)
target_link_libraries(yolo PRIVATE
	argparse::argparse
	fmt::fmt
	nlohmann_json::nlohmann_json
	spdlog::spdlog

	opencv_core
	opencv_highgui
	opencv_imgcodecs
	opencv_imgproc
	opencv_videoio

	torch
	TorchVision::TorchVision
)

add_executable(ocr ocr.cpp)
target_include_directories(ocr PRIVATE
	${onnxruntime_PREFIX}/include
)
target_link_libraries(ocr PRIVATE
	# OpenMP::OpenMP_CXX

	argparse::argparse
	Clipper2
	fmt::fmt
	mio::mio
	spdlog::spdlog

	opencv_core
	opencv_highgui
	opencv_imgcodecs
	opencv_imgproc
	opencv_videoio

	${onnxruntime_LINK_LIBRARIES}
)
