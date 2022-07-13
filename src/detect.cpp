#include <cstddef>

#include <chrono>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>
#include <fmt/chrono.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include "augmentation.hpp"
#include "inference/torchscript/yolov5.hpp"

int main(int argc, char * argv[])
{
	argparse::ArgumentParser parser;

	parser.add_argument("-i", "--input", "--image")
		.required()
		.help("Input image path for inference");
	parser.add_argument("-l", "--label")
		.required()
		.help("Corresponding list of labels for the given model.");
	parser.add_argument("-m", "-w", "--model", "--weight")
		.required()
		.help("The TorchScript model file to load.");
	parser.add_argument("-s", "--image-size")
		.required()
		.nargs(1, 2)
		.scan<'u', size_t>()
		.help("The corresponding target size of the input image for the model (either [width] [height] or [size]).");
	parser.add_argument("--conf-threshold")
		.default_value(0.3)
		.scan<'f', double>()
		.help("Confidence threshold for object detection.");
	parser.add_argument("--iou-threshold")
		.default_value(0.5)
		.scan<'f', double>()
		.help("IoU threshold for object detection.");

	parser.add_argument("-L", "--log-level")
		.scan<'u', size_t>()
		.default_value(size_t(2))
		.help("The log level (in numeric representation) of the application.");
	parser.add_argument("-S", "--show")
		.default_value(false)
		.implicit_value(true)
		.help("Do not show the image at intermediate stages.");

	parser.parse_args(argc, argv);

	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S %z (%l)] (thread %t) <%n> %v");
	spdlog::set_level(static_cast<spdlog::level::level_enum>(parser.get<size_t>("-L")));

	auto size_vec = parser.get<std::vector<size_t>>("--image-size");
	cv::Size size;
	if (size_vec.size() == 1)
		size = { size_vec[0], size_vec[0] };
	else if (size_vec.size() == 2)
		size = { size_vec[0], size_vec[1] };
	else
	[[unlikely]]
	{
		SPDLOG_ERROR("Invalid image size bypassed argument checks.");
		return 1;
	}
	SPDLOG_INFO("Target image size set to {}x{}.", size.width, size.height);

	cv::Mat image = cv::imread(parser.get<std::string>("-i"), cv::ImreadModes::IMREAD_COLOR);
	auto image_size = image.size();
	SPDLOG_INFO("Input image size is {}x{}.", image_size.width, image_size.height);

	auto [reshaped, ratio, left, right, top, bottom] = augmentation::letterbox(
		std::move(image),
		size,
		false,
		{ 114, 114, 114 }
	);
	image = std::move(reshaped);
	image_size = image.size();
	SPDLOG_INFO(
		"Reshaped image size is {}x{} (ratio {:.2f}, padding {}/{}/{}/{}).",
		image_size.width, image_size.height,
		ratio,
		left, right, top, bottom
	);

	torch::jit::FusionStrategy static_strategy { { torch::jit::FusionBehavior::STATIC, 1 } };
	torch::jit::getProfilingMode() = false;
	torch::jit::setFusionStrategy(static_strategy);
	torch::jit::setGraphExecutorOptimize(false);
	torch::jit::setTensorExprFuserEnabled(false);
	inference::torchscript::yolov5::engine model(parser.get<std::string>("-m"), parser.get<std::string>("-l"));

	auto conf_threshold = parser.get<double>("--conf-threshold"), iou_threshold = parser.get<double>("--iou-threshold");

	auto now = std::chrono::system_clock::now();
	auto ret = model.infer_image(image, conf_threshold, iou_threshold);
	SPDLOG_INFO(
		"Time elapsed for inference: {:.3}",
		std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - now)
	);

	for (const auto & [label, score, min_coord, max_coord] : ret)
	{
		SPDLOG_INFO("label {}: {:.3f}", label, score);
		cv::rectangle(image, min_coord, max_coord, { 0, 0, 255 }, 1, cv::LineTypes::LINE_AA);
	}
	cv::imshow("main", image);
	while (cv::waitKey(0) != 0x1b);

	return 0;
}
