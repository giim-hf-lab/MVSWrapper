#ifndef __INFERENCE_ONNXRUNTIME_OCR_DETECTOR_HPP__
#define __INFERENCE_ONNXRUNTIME_OCR_DETECTOR_HPP__

#include <cstddef>

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <clipper.offset.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../model.hpp"
#include "../../transformation.hpp"

namespace inference::onnxruntime::ocr
{

class detector final
{
	model _model;
	OrtAllocator * _allocator;
	cv::Scalar _mean, _stddev;
public:
	detector(
		const std::string & model_path,
		OrtAllocator * allocator,
		const cv::Scalar & mean = { 0.485, 0.456, 0.406 },
		const cv::Scalar & stddev = { 0.229, 0.224, 0.225 }
	) : _model(model_path, allocator), _allocator(allocator), _mean(mean), _stddev(stddev) {}

	[[nodiscard]]
	auto infer_image(
		cv::Mat image,
		size_t side_length,
		bool side_length_as_max,
		double score_threshold,
		bool dilation,
		bool fast_scoring,
		double box_score_threshold,
		double unclip_ratio,
		double min_box_side_length
	) &
	{
		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/preprocess_op.cpp#L34-L106
		auto scaler = inference::transformation::scale(image, side_length, side_length_as_max);
		image.convertTo(image, CV_32FC3, 1.0 / 255.0, 0.0);
		image -= _mean;
		image /= _stddev;

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/ocr_det.cpp#L119-L128
		auto image_size = image.size();
		int64_t input_shape[] { 1, 3, image_size.height, image_size.width };
		auto input_tensor = Ort::Value::CreateTensor<float>(_allocator, input_shape, 4);
		auto data_ptr = input_tensor.GetTensorMutableData<float>();
		size_t stride = image_size.height * image_size.width;
		cv::Mat split[] {
			{ image_size, CV_32FC1, data_ptr },
			{ image_size, CV_32FC1, data_ptr + stride },
			{ image_size, CV_32FC1, data_ptr + 2 * stride }
		};
		cv::split(image, split);

		int64_t output_shape[] { 1, 1, image_size.height, image_size.width };
		auto output_tensor = Ort::Value::CreateTensor<float>(_allocator, output_shape, 4);
		_model(input_tensor, output_tensor);

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/ocr_det.cpp#L132-L167
		cv::Mat scores(image_size, CV_32FC1, output_tensor.GetTensorMutableData<float>());
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
		std::vector<cv::RotatedRect> ret;
		ret.reserve(contours.size());
		// reuse allocated memory if possible
		Clipper2Lib::ClipperOffset clipper_offset;
		Clipper2Lib::Path64 contour_path;
		// #pragma omp parallel for \
		// 	schedule(dynamic) \
		// 	default(none) \
		// 	private(clipper_offset, contour_path) \
		// 	firstprivate(fast_scoring, box_score_threshold, unclip_ratio, min_box_side_length) \
		// 	shared(scaler, scores, contours, ret)
		for (size_t i = 0; i < contours.size(); ++i)
		{
			auto & contour = contours[i];
			if (contour.size() < 3)
			[[unlikely]]
				continue;

			// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/postprocess_op.cpp#L163-L207
			cv::Rect bounding;
			cv::Mat mask;
			if (fast_scoring)
			{
				auto enclosing = cv::minAreaRect(contour);
				bounding = enclosing.boundingRect();
				cv::Point2f corners[4];
				enclosing.points(corners);
				auto offset = bounding.tl();
				cv::Point points[4] {
					static_cast<cv::Point>(corners[0]) - offset,
					static_cast<cv::Point>(corners[1]) - offset,
					static_cast<cv::Point>(corners[2]) - offset,
					static_cast<cv::Point>(corners[3]) - offset
				};
				mask = cv::Mat(bounding.size(), CV_8UC1, cv::Scalar(0));
				// rectangle is always convex
				cv::fillConvexPoly(mask, points, 4, 255);
			}
			else
			{
				bounding = cv::boundingRect(contour);
				mask = cv::Mat(bounding.size(), CV_8UC1, cv::Scalar(0));
				cv::fillPoly(mask, contour, 255, cv::LineTypes::LINE_8, 0, -bounding.tl());
			}
			if (cv::mean(scores(bounding), mask)[0] < box_score_threshold)
				continue;

			// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/postprocess_op.cpp#L20-L68
			auto offset = cv::contourArea(contour, false) * unclip_ratio / cv::arcLength(contour, true);
			contour_path.clear();
			contour_path.reserve(contour.size());
			for (const auto & point : contour)
				contour_path.emplace_back(point.x, point.y);
			clipper_offset.Clear();
			clipper_offset.AddPaths({ contour_path }, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
			contour_path = std::move(clipper_offset.Execute(offset)[0]);

			contour.clear();
			contour.reserve(contour_path.size());
			for (const auto & point : contour_path)
				contour.emplace_back(point.x / scaler.ratio, point.y / scaler.ratio);

			// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/deploy/cpp_infer/src/postprocess_op.cpp#L324-L353
			auto enclosing = cv::minAreaRect(contour);
			if (auto & size = enclosing.size; std::min(size.height, size.width) <= min_box_side_length)
				continue;

			// #pragma omp critical
			ret.emplace_back(std::move(enclosing));
		}

		return ret;
	}

	void warmup(size_t height, size_t width) &
	{
		int64_t input_shape[] { 1, 3, height, width };
		auto input_tensor = Ort::Value::CreateTensor<float>(_allocator, input_shape, 4);
		int64_t output_shape[] { 1, 1, height, width };
		auto output_tensor = Ort::Value::CreateTensor<float>(_allocator, output_shape, 4);
		_model(input_tensor, output_tensor);
	}
};

}

#endif
