#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>

#include "inference/labels.hpp"

int main(int argc, char * argv[])
{
	argparse::ArgumentParser parser;

	parser.add_argument("-i", "--input", "--image-path")
		.required()
		.help("Input image path for the inference engine.");
	parser.add_argument("-l", "--labels-path")
		.required()
		.help("Corresponding list of labels for the given model.");
	parser.add_argument("-m", "-w", "--model", "--weight")
		.required()
		.help("The TorchScript model file to load.");
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

	auto labels_mapper = inference::labels_mapper::from_file(parser.get<std::string>("-l"));
	SPDLOG_INFO("Loaded {} labels.", labels_mapper.size());

	return 0;
}
