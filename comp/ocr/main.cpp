#include <exception>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifdef __INTELLISENSE__

#include "./ocr.hpp"

#else

import ocr;

#endif

int main()
{
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S %z (%l)] (thread %t) <%n> %v");
	spdlog::set_default_logger(spdlog::create<spdlog::sinks::stdout_color_sink_mt>("main"));

	std::vector<cv::Mat> images;
	if (std::filesystem::path images_path("images/samples"); std::filesystem::is_directory(images_path))
	{
		for (const auto& entry : std::filesystem::directory_iterator(images_path))
			if (auto image = cv::imread(entry.path().string(), cv::ImreadModes::IMREAD_COLOR); not image.empty())
				images.push_back(std::move(image));
	}
	else
	{
		SPDLOG_ERROR("Images directory not found.");
		return EXIT_FAILURE;
	}

	try
	{
		auto ocr = ocr::system {
			ocr::scalers::zoom { 960, false },
			ocr::detectors::db {
				{ 0.485, 0.456, 0.406 }, // { 0.48109378172549, 0.45752457890196, 0.40787054090196 },
				{ 0.229, 0.224, 0.225 }, // cv::Scalar::all(1),
				0.3,
				true,
				ocr::detectors::db::scoring_methods::ORIGINAL,
				1000,
				0.6,
				1.5,
				0.6,
				L"models/ch_PP-OCRv3_det_infer.onnx"
			},
			ocr::classifiers::concrete {
				cv::Scalar::all(0.5),
				cv::Scalar::all(0.5),
				6,
				{ 192, 48 },
				0.9,
				L"models/ch_ppocr_mobile_v2.0_cls_infer.onnx"
			},
			ocr::recognisers::ctc {
				cv::Scalar::all(0.5),
				cv::Scalar::all(0.5),
				6,
				{ 320, 48 },
				0.5,
				"models/ppocr_keys_v1.txt",
				L"models/ch_PP-OCRv3_rec_infer.onnx"
			}
		};
		for (auto& image : images)
		{
			std::vector<cv::Mat> contours;
			for (const auto& [vertices, text, score] : ocr(image))
			{
				SPDLOG_INFO("Result: ({:.3f}) {}", score, text);
				contours.push_back(vertices);
			}
			cv::polylines(image, contours, true, CV_RGB(255, 0, 0), 2, cv::LineTypes::LINE_AA);
			cv::imshow("Result", image);
			if (cv::waitKey() == 0x1b)
				return 0;
		}
	}
	catch (const std::exception& e)
	{
		SPDLOG_ERROR("Current image threw exception: {}", e.what());
		return 1;
	}

	return 0;
}
