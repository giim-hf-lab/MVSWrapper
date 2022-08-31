#ifndef __INFERENCE_PADDLE_OCR_HPP__
#define __INFERENCE_PADDLE_OCR_HPP__

#include <cmath>
#include <cstddef>

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/compile.h>
#include <fmt/core.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <paddle_api.h>
#include <paddle_inference_api.h>

namespace inference::paddlepaddle::ocr
{

class detector final
{
	[[nodiscard]]
	[[using gnu : always_inline]]
	inline static auto init_model(const std::string_view & model_dir, const std::string_view & name)
	{
		paddle_infer::Config config(
			fmt::format(FMT_COMPILE("{}/{}.pdmodel"), model_dir, name),
			fmt::format(FMT_COMPILE("{}/{}.pdiparams"), model_dir, name)
		);
		config.DisableGlogInfo();
		config.EnableMemoryOptim();
		config.EnableUseGpu(512, 0);
		config.SwitchIrOptim(true);
		config.SwitchUseFeedFetchOps(false);
		config.SwitchSpecifyInputNames(true);

		return paddle_infer::Predictor(config);
	}

	paddle_infer::Predictor _predictor;
	std::vector<std::string> _input_names, _output_names;
	cv::Scalar _mean, _stddev;
public:
	detector(
		const std::string_view & model_dir,
		const std::string_view & name = "inference",
		const cv::Scalar & mean = { 0.485, 0.456, 0.406 },
		const cv::Scalar & stddev = { 0.229, 0.224, 0.225 }
	) :
		_predictor(init_model(model_dir, name)),
		_input_names(_predictor.GetInputNames()),
		_output_names(_predictor.GetOutputNames()),
		_mean(mean),
		_stddev(stddev)
	{
		if (_input_names.size() > 1)
		[[unlikely]]
			throw std::runtime_error("OCR detector model requires too many inputs");

		if (_output_names.size() > 1)
		[[unlikely]]
			throw std::runtime_error("OCR detector model yields too many outputs");
	}

	[[nodiscard]]
	auto infer_image(
		cv::Mat image,
		double score_threshold,
		bool dilation,
		bool fast_scoring,
		double box_score_threshold,
		double enlarge_ratio
	) &
	{
		auto image_size = image.size();
		image.convertTo(image, CV_32FC3, 1.0 / 255.0, 0.0);
		image -= _mean;
		image /= _stddev;

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/ocr_det.cpp#L119-L128
		static constexpr int swap_indices[] {
			2, 0,
			0, 1,
			1, 2
		};
		size_t stride = image_size.height * image_size.width;
		std::vector<float> buffer(3 * stride);
		cv::Mat split[] {
			{ image_size.height, image_size.width, CV_32FC1, buffer.data() },
			{ image_size.height, image_size.width, CV_32FC1, buffer.data() + stride },
			{ image_size.height, image_size.width, CV_32FC1, buffer.data() + 2 * stride }
		};
		cv::mixChannels(&image, 1, split, 3, swap_indices, 3);
		auto tensor = _predictor.GetInputHandle(_input_names[0]);
		tensor->Reshape({ 1, 3, image_size.height, image_size.width });
		tensor->CopyFromCpu(buffer.data());

		_predictor.Run();

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/ocr_det.cpp#L132-L167
		tensor = _predictor.GetOutputHandle(_output_names[0]);
		if (auto shape = tensor->shape(); shape.size() != 4)
		[[unlikely]]
			throw std::runtime_error("OCR detector model yields invalid output dimension");
		else if (shape[0] != 1 || shape[1] != 1 || shape[2] != image_size.height || shape[3] != image_size.width)
		[[unlikely]]
			throw std::runtime_error("OCR detector model yields invalid output shape");
		cv::Mat scores(image_size, CV_32FC1);
		tensor->CopyToCpu(scores.ptr<float>());
		cv::Mat bitmap;
		cv::threshold(scores, bitmap, score_threshold, 255, cv::ThresholdTypes::THRESH_BINARY);
		bitmap.convertTo(bitmap, CV_8UC1, 1.0, 0.0);
		if (dilation)
			cv::dilate(bitmap, bitmap, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, { 2, 2 }));

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/postprocess_op.cpp#L247-L322
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(
			bitmap,
			contours,
			cv::RetrievalModes::RETR_LIST,
			cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE
		);
		size_t valid_count = 0;
		enlarge_ratio = sqrt(enlarge_ratio);
		for (size_t i = 0; i < contours.size(); ++i)
		{
			auto & contour = contours[i];
			if (contour.size() < 3)
			[[unlikely]]
				continue;

			// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/postprocess_op.cpp#L163-L207
			cv::RotatedRect enclosing;
			if (fast_scoring)
			{
				enclosing = cv::minAreaRect(contour);
				auto bounding = enclosing.boundingRect();
				cv::Point2f corners[4];
				enclosing.points(corners);
				auto offset = bounding.tl();
				cv::Point points[4] {
					static_cast<cv::Point>(corners[0]) - offset,
					static_cast<cv::Point>(corners[1]) - offset,
					static_cast<cv::Point>(corners[2]) - offset,
					static_cast<cv::Point>(corners[3]) - offset
				};
				cv::Mat mask(bounding.size(), CV_8UC1, cv::Scalar(0));
				// rectangle is always convex
				cv::fillConvexPoly(mask, points, 4, 255);
				if (cv::mean(scores(bounding), mask)[0] < box_score_threshold)
					continue;
			}
			else
			{
				auto bounding = cv::boundingRect(contour);
				cv::Mat mask(bounding.size(), CV_8UC1, cv::Scalar(0));
				cv::fillPoly(mask, contour, 255, cv::LineTypes::LINE_8, 0, -bounding.tl());
				if (cv::mean(scores(bounding), mask)[0] < box_score_threshold)
					continue;
				enclosing = cv::minAreaRect(contour);
			}

			// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/postprocess_op.cpp#L20-L68
			auto enlarge_distance = std::max(enclosing.size.height, enclosing.size.width) * (enlarge_ratio - 1);
			enclosing.size.height += enlarge_distance;
			enclosing.size.width += enlarge_distance;
			cv::Point2f corners[4];
			enclosing.points(corners);
			contour.clear();
			contour.reserve(4);
			for (auto & corner : corners)
				contour.emplace_back(
					std::clamp(int(corner.x), 0, image_size.width - 1),
					std::clamp(int(corner.y), 0, image_size.height - 1)
				);

			if (i != valid_count)
				contours[valid_count] = std::move(contour);
			++valid_count;
		}
		contours.resize(valid_count);
		contours.shrink_to_fit();
		return contours;
	}
};

}

#endif
