#include <cstddef>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>
#include <fmt/chrono.h>
// #include <omp.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "inference/onnxruntime/ocr/detector.hpp"

int main(int argc, char * argv[])
{
	using namespace std::string_literals;

	argparse::ArgumentParser parser;

	parser.add_argument("-i", "--images")
		.nargs(argparse::nargs_pattern::at_least_one)
		.append()
		.help("Input images to the inference engine.");
	parser.add_argument("--det-model-path")
		.required()
		.help("Directory containing the detection model.");

	parser.add_argument("-L", "--log-level")
		.scan<'u', size_t>()
		.default_value(size_t(2))
		.help("The log level (in numeric representation) of the application.");

	parser.parse_args(argc, argv);

	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S %z (%l)] (thread %t) <%n> %v");
	spdlog::set_level(static_cast<spdlog::level::level_enum>(parser.get<size_t>("-L")));

	auto det_model_path = parser.get<std::string>("--det-model-path");

	// #pragma omp parallel for
	// for (size_t i = 0; i < omp_get_num_procs(); ++i);

	Ort::AllocatorWithDefaultOptions allocator;
	inference::onnxruntime::ocr::detector detector(det_model_path, allocator);
	detector.warmup(640, 960);

	for (const auto & image_path : parser.get<std::vector<std::string>>("-i"))
	{
		if (!std::filesystem::exists(image_path))
		[[unlikely]]
		{
			SPDLOG_ERROR("Image {} does not exist.", image_path);
			continue;
		}

		SPDLOG_INFO("Processing image {}.", image_path);

		auto image = cv::imread(image_path);
		auto now = std::chrono::system_clock::now();
		auto detector_results = detector.infer_image(image, 960, true, 0.3, true, false, 0.6, 1.5, 4);
		SPDLOG_INFO(
			"->  detector time is {:.3}.",
			std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - now)
		);

		now = std::chrono::system_clock::now();
		std::vector<cv::Mat> cropped_images;
		cropped_images.reserve(detector_results.size());
		for (const auto & detector_result : detector_results)
		{
			// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/utility.cpp#L101-L154
			auto & cropped_image = cropped_images.emplace_back();
			auto & size = detector_result.size;
			cv::Point2f vertices[4], upright_vertices[4] {
				{ 0, size.height },
				{ 0, 0 },
				{ size.width, 0 },
				{ size.width, size.height }
			};
			detector_result.points(vertices);
			cv::warpPerspective(
				image,
				cropped_image,
				cv::getPerspectiveTransform(vertices, upright_vertices),
				size
			);
			if (size.height > size.width * 1.5)
				cv::rotate(cropped_image, cropped_image, cv::RotateFlags::ROTATE_90_CLOCKWISE);
		}
		SPDLOG_INFO(
			"->  detector result processing time is {:.3}.",
			std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - now)
		);
	}

	return 0;
}
