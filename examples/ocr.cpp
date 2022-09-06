#include <cstddef>

#include <chrono>
#include <filesystem>
#include <string>
#include <utility>
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

#include "inference/onnxruntime/ocr/classifier.hpp"
#include "inference/onnxruntime/ocr/detector.hpp"
#include "inference/onnxruntime/ocr/recogniser.hpp"

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
		.help("Path to the detector model.");
	parser.add_argument("--cls-model-path")
		.required()
		.help("Path to the classifier model.");
	parser.add_argument("--rec-model-path")
		.required()
		.help("Path to the recogniser model.");
	parser.add_argument("--rec-dict-path")
		.required()
		.help("Path to the recogniser dictionary.");

	parser.add_argument("--det-side-length")
		.default_value(size_t(960))
		.scan<'u', size_t>()
		.help("The maximum side length of the detector input.");
	parser.add_argument("--det-side-length-as-min")
		.default_value(false)
		.implicit_value(true)
		.help("Whether to use the side length as the minimum side length of the detector input.");
	parser.add_argument("--det-threshold")
		.default_value(0.3)
		.scan<'f', double>()
		.help("Threshold for the detector.");
	parser.add_argument("--det-no-dilation")
		.default_value(false)
		.implicit_value(true)
		.help("Whether to disable dilation for the detector.");
	parser.add_argument("--det-fast-scoring")
		.default_value(false)
		.implicit_value(true)
		.help("Whether to use the fast scoring method for the detected polygons.");
	parser.add_argument("--det-score-threshold")
		.default_value(0.6)
		.scan<'f', double>()
		.help("Threshold for the scores of detected polygons.");
	parser.add_argument("--det-unclip-ratio")
		.default_value(1.5)
		.scan<'f', double>()
		.help("Unclip ratio of the detected polygons.");
	parser.add_argument("--det-box-min-side-length")
		.default_value(size_t(4))
		.scan<'u', size_t>()
		.help("Minimum side length of the detected boxes.");

	parser.add_argument("--cls-shape")
		.default_value(std::vector<size_t> { 48, 192 })
		.nargs(2)
		.scan<'u', size_t>()
		.help("Shape of the classifier input ([height, width]).");
	parser.add_argument("--cls-batch")
		.default_value(size_t(10))
		.scan<'u', size_t>()
		.help("Batch size of the classifier.");
	parser.add_argument("--cls-threshold")
		.default_value(0.9)
		.scan<'f', double>()
		.help("Threshold for the classifier.");

	parser.add_argument("--rec-shape")
		.default_value(std::vector<size_t> { 48, 320 })
		.nargs(2)
		.scan<'u', size_t>()
		.help("Shape of the recogniser input ([height, width]).");
	parser.add_argument("--rec-batch")
		.default_value(size_t(10))
		.scan<'u', size_t>()
		.help("Batch size of the recogniser.");
	parser.add_argument("--rec-threshold")
		.default_value(0.5)
		.scan<'f', double>()
		.help("Threshold for the recogniser.");

	parser.add_argument("-L", "--log-level")
		.scan<'u', size_t>()
		.default_value(size_t(2))
		.help("The log level (in numeric representation) of the application.");

	parser.parse_args(argc, argv);

	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S %z (%l)] (thread %t) <%n> %v");
	spdlog::set_level(static_cast<spdlog::level::level_enum>(parser.get<size_t>("-L")));

	// #pragma omp parallel for
	// for (size_t i = 0; i < omp_get_num_procs(); ++i);

	Ort::AllocatorWithDefaultOptions allocator;

	inference::onnxruntime::ocr::detector detector(parser.get<std::string>("--det-model-path"), allocator);
	auto det_side_length = parser.get<size_t>("--det-side-length");
	auto det_side_length_as_max = not parser.get<bool>("--det-side-length-as-min");
	auto det_threshold = parser.get<double>("--det-threshold");
	auto det_dilation = not parser.get<bool>("--det-no-dilation");
	auto det_fast_scoring = parser.get<bool>("--det-fast-scoring");
	auto det_score_threshold = parser.get<double>("--det-score-threshold");
	auto det_unclip_ratio = parser.get<double>("--det-unclip-ratio");
	auto det_box_min_side_length = parser.get<size_t>("--det-box-min-side-length");

	inference::onnxruntime::ocr::classifier classifier(parser.get<std::string>("--cls-model-path"), allocator);
	auto cls_shape_vec = parser.get<std::vector<size_t>>("--cls-shape");
	auto cls_batch = parser.get<size_t>("--cls-batch");
	auto cls_threshold = parser.get<double>("--cls-threshold");

	inference::onnxruntime::ocr::recogniser recogniser(
		parser.get<std::string>("--rec-model-path"),
		allocator,
		parser.get<std::string>("--rec-dict-path")
	);
	auto rec_shape_vec = parser.get<std::vector<size_t>>("--rec-shape");
	auto rec_batch = parser.get<size_t>("--rec-batch");
	auto rec_threshold = parser.get<double>("--rec-threshold");

	if (cls_shape_vec.size() != 2)
	[[unlikely]]
	{
		SPDLOG_ERROR("Invalid shape of the classifier input.");
		return 1;
	}
	cv::Size cls_shape(cls_shape_vec[1], cls_shape_vec[0]);

	if (rec_shape_vec.size() != 2)
	[[unlikely]]
	{
		SPDLOG_ERROR("Invalid shape of the recogniser input.");
		return 1;
	}
	cv::Size rec_shape(rec_shape_vec[1], rec_shape_vec[0]);

	for (const auto & image_path : parser.get<std::vector<std::string>>("-i"))
	{
		if (not std::filesystem::exists(image_path))
		[[unlikely]]
		{
			SPDLOG_ERROR("Image {} does not exist.", image_path);
			continue;
		}

		SPDLOG_INFO("Processing image {}.", image_path);

		auto image = cv::imread(image_path);
		detector.warmup(640, 960);

		auto now = std::chrono::system_clock::now();
		auto detector_results = detector.forward(
			image,
			det_side_length,
			det_side_length_as_max,
			det_threshold,
			det_dilation,
			det_fast_scoring,
			det_score_threshold,
			det_unclip_ratio,
			det_box_min_side_length
		);
		SPDLOG_INFO(
			"->  detector time is {:.3}.",
			std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - now)
		);

		classifier.warmup(cls_batch, cls_shape);

		now = std::chrono::system_clock::now();
		auto classifier_results = classifier.forward(
			image,
			detector_results,
			cls_batch,
			cls_shape,
			cls_threshold
		);
		SPDLOG_INFO(
			"->  classifier time is {:.3}.",
			std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - now)
		);

		recogniser.warmup(rec_batch, rec_shape);

		now = std::chrono::system_clock::now();
		auto recogniser_results = recogniser.forward(
			classifier_results,
			rec_batch,
			rec_shape,
			rec_threshold
		);
		SPDLOG_INFO(
			"->  recogniser time is {:.3}.",
			std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - now)
		);

		for (const auto & [index, str, score] : recogniser_results)
			SPDLOG_INFO("    ->  {}: {} ({:.3}).", index, str, score);
	}

	return 0;
}
