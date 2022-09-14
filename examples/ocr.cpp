#include <cstddef>

#include <chrono>
#include <filesystem>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <argparse/argparse.hpp>
#include <fmt/chrono.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <inferences/framework/onnxruntime/ocr/classifier.hpp>
#include <inferences/framework/onnxruntime/ocr/detector.hpp>
#include <inferences/framework/onnxruntime/ocr/recogniser.hpp>

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

	parser.add_argument("--det-shape")
		.default_value(std::vector<size_t> { 960, 960 })
		.nargs(2)
		.scan<'u', size_t>()
		.help("Shape of the detector input ([height, width]).");
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

	parser.add_argument("--cls-batch-size")
		.default_value(size_t(10))
		.scan<'u', size_t>()
		.help("Batch size of the classifier.");
	parser.add_argument("--cls-shape")
		.default_value(std::vector<size_t> { 48, 192 })
		.nargs(2)
		.scan<'u', size_t>()
		.help("Shape of the classifier input ([height, width]).");
	parser.add_argument("--cls-threshold")
		.default_value(0.9)
		.scan<'f', double>()
		.help("Threshold for the classifier.");

	parser.add_argument("--rec-batch-size")
		.default_value(size_t(10))
		.scan<'u', size_t>()
		.help("Batch size of the recogniser.");
	parser.add_argument("--rec-shape")
		.default_value(std::vector<size_t> { 48, 320 })
		.nargs(2)
		.scan<'u', size_t>()
		.help("Shape of the recogniser input ([height, width]).");
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

	auto det_shape_vec = parser.get<std::vector<size_t>>("--det-shape");
	if (det_shape_vec.size() != 2)
	[[unlikely]]
	{
		SPDLOG_ERROR("Invalid shape of the classifier input.");
		return 1;
	}

	auto cls_shape_vec = parser.get<std::vector<size_t>>("--cls-shape");
	if (cls_shape_vec.size() != 2)
	[[unlikely]]
	{
		SPDLOG_ERROR("Invalid shape of the classifier input.");
		return 1;
	}

	auto rec_shape_vec = parser.get<std::vector<size_t>>("--rec-shape");
	if (rec_shape_vec.size() != 2)
	[[unlikely]]
	{
		SPDLOG_ERROR("Invalid shape of the recogniser input.");
		return 1;
	}

	static const auto IMAGE_NET_MEAN = CV_RGB(0.485, 0.456, 0.406);
	static const auto IMAGE_NET_STDDEV = CV_RGB(0.229, 0.224, 0.225);

	auto detector_model_path = parser.get<std::string>("--det-model-path");
	inferences::framework::onnxruntime::ocr::detector detector(detector_model_path);
	SPDLOG_INFO("Detector model {} loaded.", detector_model_path);

	inferences::framework::onnxruntime::ocr::detector::parameters detector_parameters(
		{ det_shape_vec[1], det_shape_vec[0] },
		IMAGE_NET_MEAN,
		IMAGE_NET_STDDEV,
		parser.get<double>("--det-threshold"),
		not parser.get<bool>("--det-no-dilation"),
		parser.get<bool>("--det-fast-scoring"),
		parser.get<double>("--det-score-threshold"),
		parser.get<double>("--det-unclip-ratio"),
		parser.get<size_t>("--det-box-min-side-length")
	);

	auto classifier_model_path = parser.get<std::string>("--cls-model-path");
	inferences::framework::onnxruntime::ocr::classifier classifier(classifier_model_path);
	SPDLOG_INFO("Classifier model {} loaded.", classifier_model_path);

	inferences::framework::onnxruntime::ocr::classifier::parameters classifier_parameters(
		parser.get<size_t>("--cls-batch-size"),
		{ cls_shape_vec[1], cls_shape_vec[0] },
		IMAGE_NET_MEAN,
		IMAGE_NET_STDDEV,
		parser.get<double>("--cls-threshold")
	);

	auto recogniser_model_path = parser.get<std::string>("--rec-model-path");
	auto recogniser_dict_path = parser.get<std::string>("--rec-dict-path");
	inferences::framework::onnxruntime::ocr::recogniser recogniser(recogniser_model_path, recogniser_dict_path);
	SPDLOG_INFO("Recogniser model {} loaded.", recogniser_model_path);

	inferences::framework::onnxruntime::ocr::recogniser::parameters recogniser_parameters(
		parser.get<size_t>("--rec-batch-size"),
		{ rec_shape_vec[1], rec_shape_vec[0] },
		IMAGE_NET_MEAN,
		IMAGE_NET_STDDEV,
		parser.get<double>("--rec-threshold")
	);

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

		auto rectangles = detector(image, detector_parameters);

		auto fragments = classifier(image, rectangles, classifier_parameters);

		auto results = recogniser(fragments, recogniser_parameters);

		std::vector<std::vector<cv::Point>> contours;
		contours.reserve(results.size());
		for (const auto& [index, label, score] : results)
		{
			cv::Point2f vertices[4];
			rectangles[index].points(vertices);
			contours.emplace_back(vertices, vertices + 4);

			SPDLOG_INFO("    ->  ({}) {}", score, label);
		}

		cv::polylines(image, contours, true, cv::Scalar(0, 0, 255), 2, cv::LineTypes::LINE_AA);

		cv::imshow("image", image);
		while (cv::waitKey(0) != 0x1b);
	}

	return 0;
}
