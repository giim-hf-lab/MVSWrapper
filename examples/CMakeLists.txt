add_executable(detect detect.cpp)
target_link_libraries(detect PRIVATE
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
target_link_libraries(ocr PRIVATE
	argparse::argparse
	fmt::fmt
	spdlog::spdlog

	paddle::inference::static
)
