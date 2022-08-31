#include <cstddef>

#include <string>
#include <utility>
#include <vector>

#include <argparse/argparse.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "inference/paddlepaddle/ocr.hpp"
#include "inference/transformation.hpp"

int main(int argc, char * argv[])
{
	using namespace std::string_literals;

	argparse::ArgumentParser parser;

	parser.add_argument("-i", "--images")
		.nargs(argparse::nargs_pattern::at_least_one)
		.append()
		.help("Input images to the inference engine.");
	parser.add_argument("--det-model-dir")
		.required()
		.help("Directory containing the detection model.");
	parser.add_argument("--det-model-name")
		.default_value("inference"s)
		.help("Name of the detection model.");

	parser.add_argument("-L", "--log-level")
		.scan<'u', size_t>()
		.default_value(size_t(2))
		.help("The log level (in numeric representation) of the application.");

	parser.parse_args(argc, argv);

	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S %z (%l)] (thread %t) <%n> %v");
	spdlog::set_level(static_cast<spdlog::level::level_enum>(parser.get<size_t>("-L")));

	auto det_model_dir = parser.get<std::string>("--det-model-dir");
	auto det_model_name = parser.get<std::string>("--det-model-name");

	inference::paddlepaddle::ocr::detector detector(det_model_dir, det_model_name);

	const cv::Scalar mean { 0.485, 0.456, 0.406 }, stddev { 0.229, 0.224, 0.225 };
	for (const auto & image_path : parser.get<std::vector<std::string>>("-i"))
	{
		auto image = cv::imread(image_path);
		auto scaler = inference::transformation::scale(image, 960, true);
		cv::polylines(
			image,
			detector.infer_image(image, 0.3, true, true, 0.6, 1.5),
			true,
			CV_RGB(255, 0, 0),
			2,
			cv::LineTypes::LINE_AA
		);
		cv::imshow("image", image);
		while (cv::waitKey(0) != 0x1b);
	}

	return 0;
}
