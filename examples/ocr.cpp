#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>

#include <inference/paddlepaddle/ocr.hpp>

int main(int argc, char * argv[])
{
	argparse::ArgumentParser parser;

	parser.add_argument("-i", "--images")
		.nargs(argparse::nargs_pattern::at_least_one)
		.append()
		.help("Input images to the inference engine.");

	parser.add_argument("-L", "--log-level")
		.scan<'u', size_t>()
		.default_value(size_t(2))
		.help("The log level (in numeric representation) of the application.");

	parser.parse_args(argc, argv);

	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S %z (%l)] (thread %t) <%n> %v");
	spdlog::set_level(static_cast<spdlog::level::level_enum>(parser.get<size_t>("-L")));

	return 0;
}
