#include <cstddef>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>
#include <fmt/chrono.h>
#include <fmt/ranges.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <inferences/framework/torchscript/yolo/v5.hpp>

int main(int argc, char * argv[])
{
	argparse::ArgumentParser parser;

	parser.add_argument("-i", "--images")
		.nargs(argparse::nargs_pattern::at_least_one)
		.append()
		.help("Input images to the inference engine.");
	parser.add_argument("-I", "--included-labels")
		.nargs(argparse::nargs_pattern::at_least_one)
		.append()
		.help("The list of labels to be included in the output.");
	parser.add_argument("-v", "--videos")
		.nargs(argparse::nargs_pattern::at_least_one)
		.append()
		.help("The videos to be analysed (either files/URLs).");
	parser.add_argument("-w", "--weight")
		.required()
		.help("The TorchScript model file to load.");
	parser.add_argument("-X", "--excluded-labels")
		.nargs(argparse::nargs_pattern::at_least_one)
		.append()
		.help("The list of labels to be excluded in the output.");
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

	parser.parse_args(argc, argv);

	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S %z (%l)] (thread %t) <%n> %v");
	spdlog::set_level(static_cast<spdlog::level::level_enum>(parser.get<size_t>("-L")));

	auto included_labels = parser.get<std::vector<std::string>>("-I");
	SPDLOG_INFO("Included labels: [ {} ].", fmt::join(included_labels, " , "));

	auto excluded_labels = parser.get<std::vector<std::string>>("-X");
	SPDLOG_INFO("Excluded labels: [ {} ].", fmt::join(excluded_labels, " , "));

	inferences::framework::torchscript::yolo::v5::parameters parameters(
		false,
		parser.get<double>("--conf-threshold"),
		parser.get<double>("--iou-threshold")
	);

	torch::jit::FusionStrategy static_strategy { { torch::jit::FusionBehavior::STATIC, 1 } };
	torch::jit::getProfilingMode() = false;
	torch::jit::setFusionStrategy(static_strategy);
	torch::jit::setGraphExecutorOptimize(false);
	torch::jit::setTensorExprFuserEnabled(false);
	auto model_path = parser.get<std::string>("-w");
	inferences::framework::torchscript::yolo::v5 model(model_path, torch::kCUDA, torch::kFloat16);
	SPDLOG_INFO("Model {} loaded.", model_path);
	model.warmup();

	auto filter = model.create_filter(included_labels, excluded_labels);

	for (const auto & image_path : parser.get<std::vector<std::string>>("-i"))
	{
		if (!std::filesystem::exists(image_path))
		[[unlikely]]
		{
			SPDLOG_ERROR("Image {} does not exist.", image_path);
			continue;
		}

		SPDLOG_INFO("Processing image {}.", image_path);

		auto image = cv::imread(image_path, cv::ImreadModes::IMREAD_COLOR);
		auto size = image.size();
		SPDLOG_INFO("->  Image size is {}x{}.", size.width, size.height);

		auto now = std::chrono::system_clock::now();
		auto ret = model(image, parameters, filter);
		SPDLOG_INFO(
			"->  Inference time is {:.3}.",
			std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - now)
		);

		for (const auto & [label, score, min_coord, max_coord] : ret)
			cv::rectangle(image, min_coord, max_coord, { 0, 0, 255 }, 1, cv::LineTypes::LINE_AA);

		cv::imshow("main", image);
		while (cv::waitKey(0) != 0x1b);
	}

	for (const auto & video_path : parser.get<std::vector<std::string>>("-v"))
	{
		SPDLOG_INFO("Processing video {}.", video_path);

		cv::Mat frame;
		auto capture = cv::VideoCapture(video_path);
		while (capture.isOpened())
		{
			capture >> frame;
			auto ret = model(frame, parameters, filter);

			for (const auto & [label, score, min_coord, max_coord] : ret)
				cv::rectangle(frame, min_coord, max_coord, { 0, 0, 255 }, 1, cv::LineTypes::LINE_AA);

			cv::imshow("main", frame);
			if (cv::waitKey(1) == 0x1b)
			{
				SPDLOG_WARN("->  Current video processing stopped as requested.");
				break;
			}
		}

		if (cv::waitKey(0) == 0x1b)
		{
			SPDLOG_WARN("Video processing stopped as requested.");
			break;
		}
	}

	return 0;
}
