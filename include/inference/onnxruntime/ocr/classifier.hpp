#ifndef __INFERENCE_ONNXRUNTIME_OCR_CLASSIFIER_HPP__
#define __INFERENCE_ONNXRUNTIME_OCR_CLASSIFIER_HPP__

#include <cstddef>
#include <cstdint>

#include <string>
#include <utility>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../model.hpp"
#include "../../constants.hpp"
#include "../../transformation.hpp"

namespace inference::onnxruntime::ocr
{

class classifier final
{
	model _model;
	OrtAllocator * _allocator;
	cv::Scalar _mean, _stddev;
public:
	classifier(
		const std::string & model_path,
		OrtAllocator * allocator,
		const cv::Scalar & mean = cv::IMAGE_NET_MEAN,
		const cv::Scalar & stddev = cv::IMAGE_NET_STDDEV
	) : _model(model_path, allocator), _allocator(allocator), _mean(mean), _stddev(stddev) {}

	[[nodiscard]]
	auto forward(
		const cv::Mat & image,
		const std::vector<cv::RotatedRect> & detector_results,
		size_t batch,
		const cv::Size & shape,
		double threshold
	) &
	{
		std::vector<cv::Mat> results, fragments;
		results.reserve(detector_results.size());
		fragments.reserve(batch);

		int64_t input_shape[] { batch, 3, shape.height, shape.width };
		auto input_tensor = Ort::Value::CreateTensor<float>(_allocator, input_shape, 4);
		int64_t output_shape[] { batch, 2 };
		auto output_tensor = Ort::Value::CreateTensor<float>(_allocator, output_shape, 2);
		size_t stride = shape.height * shape.width;

		for (size_t i = 0; i < detector_results.size(); i += batch)
		{
			size_t left = std::min(batch, detector_results.size() - i);
			for (size_t j = 0, pos = i; j < left; ++j, ++pos)
			{
				const auto & detector_result = detector_results[pos];

				auto & cropped_image = fragments.emplace_back();
				auto & size = detector_result.size;
				cv::Point2f vertices[4], upright_vertices[4] {
					{ 0, size.height },
					{ 0, 0 },
					{ size.width, 0 },
					{ size.width, size.height }
				};
				detector_result.points(vertices);
				cv::warpPerspective(
					image,
					cropped_image,
					cv::getPerspectiveTransform(vertices, upright_vertices),
					size
				);
				if (size.height > size.width)
					cv::rotate(cropped_image, cropped_image, cv::RotateFlags::ROTATE_90_CLOCKWISE);
			}

			auto input_ptr = input_tensor.GetTensorMutableData<float>();
			for (const auto & fragment : fragments)
			{
				cv::Mat transformed;
				transformation::scale_letterbox(fragment, transformed, shape, _mean, _stddev);

				cv::Mat split[] {
					{ shape, CV_32FC1, input_ptr },
					{ shape, CV_32FC1, input_ptr + stride },
					{ shape, CV_32FC1, input_ptr + 2 * stride }
				};
				cv::split(transformed, split);

				input_ptr += 3 * stride;
			}

			_model(input_tensor, output_tensor);

			auto output_ptr = output_tensor.GetTensorData<float>();
			for (size_t j = 0, pos = 0; j < left; ++j, pos += 2)
				if (auto s1 = output_ptr[pos], s2 = output_ptr[pos + 1]; s1 < s2 and s2 > threshold)
					cv::rotate(fragments[j], results.emplace_back(), cv::RotateFlags::ROTATE_180);
				else
					results.emplace_back(std::move(fragments[j]));

			fragments.clear();
		}

		return results;
	}

	void warmup(size_t batch, const cv::Size & shape) &
	{
		int64_t input_shape[] { batch, 3, shape.height, shape.width };
		auto input_tensor = Ort::Value::CreateTensor<float>(_allocator, input_shape, 4);
		int64_t output_shape[] { batch, 2 };
		auto output_tensor = Ort::Value::CreateTensor<float>(_allocator, output_shape, 2);
		_model(input_tensor, output_tensor);
	}
};

}

#endif
