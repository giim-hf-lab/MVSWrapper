add_executable(detect detect.cpp)
target_include_directories(detect PRIVATE
	${TORCH_INCLUDE_DIRS}
	${TorchVision_INCLUDE_DIR}
)
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

	${TORCH_LIBRARIES}
	TorchVision::TorchVision
)
