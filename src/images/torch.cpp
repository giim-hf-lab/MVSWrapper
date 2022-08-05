#include <cstddef>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>
#include <fmt/chrono.h>
#include <fmt/ranges.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <spdlog/spdlog.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include "inference/torchscript/yolo/v5.hpp"
#include "inference/labels.hpp"

int main(int argc, char * argv[])
{
	argparse::ArgumentParser parser;

	parser.add_argument("-i", "--input", "--image-path")
		.required()
		.help("Input image path for the inference engine.");
	parser.add_argument("-I", "--included-labels")
		.nargs(argparse::nargs_pattern::at_least_one)
		.append()
		.help("The list of labels to be included in the output.");
	parser.add_argument("-l", "--labels-path")
		.required()
		.help("Corresponding list of labels for the given model.");
	parser.add_argument("-m", "-w", "--model", "--weight")
		.required()
		.help("The TorchScript model file to load.");
	parser.add_argument("-s", "--accepted-size")
		.required()
		.nargs(1, 2)
		.scan<'u', size_t>()
		.help("The accepted target size of the model input (either [width] [height] or [size]).");
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

	auto size_vec = parser.get<std::vector<size_t>>("-s");
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
	SPDLOG_INFO("Accepted image size set to {}x{}.", size.width, size.height);

	auto labels_mapper = inference::labels_mapper::from_file(parser.get<std::string>("-l"));
	SPDLOG_INFO("Loaded {} labels.", labels_mapper.size());

	auto included_labels = parser.get<std::vector<std::string>>("-I");
	SPDLOG_INFO("Included labels: [ {} ].", fmt::join(included_labels, " , "));
	std::vector<int64_t> included_indices;
	if (auto size = included_labels.size(); size)
	{
		included_indices.reserve(included_labels.size());
		for (const auto & label : included_labels)
			included_indices.emplace_back(labels_mapper.at(label));
		std::ranges::sort(included_indices);
		auto trim_ranges = std::ranges::unique(included_indices);
		included_indices.erase(trim_ranges.begin(), trim_ranges.end());
	}
	SPDLOG_INFO("Included labels (indices): [ {} ].", fmt::join(included_indices, " , "));

	auto excluded_labels = parser.get<std::vector<std::string>>("-X");
	SPDLOG_INFO("Excluded labels: [ {} ].", fmt::join(excluded_labels, " , "));
	std::vector<int64_t> excluded_indices;
	if (auto size = excluded_labels.size(); size)
	{
		excluded_indices.reserve(excluded_labels.size());
		for (const auto & label : excluded_labels)
			excluded_indices.emplace_back(labels_mapper.at(label));
		std::ranges::sort(excluded_indices);
		auto trim_ranges = std::ranges::unique(excluded_indices);
		excluded_indices.erase(trim_ranges.begin(), trim_ranges.end());
	}
	SPDLOG_INFO("Excluded labels (indices): [ {} ].", fmt::join(excluded_indices, " , "));

	torch::jit::FusionStrategy static_strategy { { torch::jit::FusionBehavior::STATIC, 1 } };
	torch::jit::getProfilingMode() = false;
	torch::jit::setFusionStrategy(static_strategy);
	torch::jit::setGraphExecutorOptimize(false);
	torch::jit::setTensorExprFuserEnabled(false);
	inference::torchscript::yolo::v5 model(
		parser.get<std::string>("-m"),
		torch::kCUDA,
		torch::kFloat16,
		std::move(size),
		false,
		// https://github.com/ultralytics/yolov5/blob/v6.1/utils/augmentations.py#L91
		{ 114, 114, 114 }
	);

	cv::Mat image = cv::imread(parser.get<std::string>("-i"), cv::ImreadModes::IMREAD_COLOR);
	size = image.size();
	SPDLOG_INFO("Input image size is {}x{}.", size.width, size.height);

	auto conf_threshold = parser.get<double>("--conf-threshold"), iou_threshold = parser.get<double>("--iou-threshold");
	SPDLOG_INFO("Thresholds set to conf={:.3f} and iou={:.3f}.", conf_threshold, iou_threshold);

	auto now = std::chrono::system_clock::now();
	auto ret = model.infer_image(image, conf_threshold, iou_threshold, included_indices, excluded_indices);
	SPDLOG_INFO(
		"Time elapsed for inference: {:.3}",
		std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - now)
	);

	SPDLOG_INFO("{} results detected", ret.size());

	for (const auto & [label, score, min_coord, max_coord] : ret)
	{
		SPDLOG_INFO(
			"label {}: {:.3f} [[{}, {}], [{}, {}]]",
			labels_mapper.at(label),
			score,
			min_coord.x,
			min_coord.y,
			max_coord.x,
			max_coord.y
		);
		cv::rectangle(image, min_coord, max_coord, { 0, 0, 255 }, 1, cv::LineTypes::LINE_AA);
	}
	cv::imshow("main", image);
	while (cv::waitKey(0) != 0x1b);

	return 0;
}
