#ifndef __INFERENCE_ONNXRUNTIME_OCR_RECOGNISER_HPP__
#define __INFERENCE_ONNXRUNTIME_OCR_RECOGNISER_HPP__

#include <cstddef>
#include <cstdint>

#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <mio/mmap.hpp>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../model.hpp"
#include "../../constants.hpp"
#include "../../transformation.hpp"

namespace inference::onnxruntime::ocr
{

class recogniser final
{
	model _model;
	OrtAllocator * _allocator;
	std::vector<std::vector<char>> _dictionary;
	cv::Scalar _mean, _stddev;
public:
	recogniser(
		const std::string & model_path,
		OrtAllocator * allocator,
		const std::string & dict_path,
		const cv::Scalar & mean = cv::IMAGE_NET_MEAN,
		const cv::Scalar & stddev = cv::IMAGE_NET_STDDEV
	) : _model(model_path, allocator), _allocator(allocator), _dictionary(), _mean(mean), _stddev(stddev)
	{
		_dictionary.emplace_back();

		mio::mmap_source dict(dict_path);
		std::vector<char> buffer;
		buffer.reserve(4);
		for (auto c : std::string_view(dict.data(), dict.size()))
			switch (c)
			{
				case '\r':
				case '\n':
					if (buffer.size())
					[[likely]]
					{
						_dictionary.push_back(buffer);
						buffer.clear();
					}
					break;
				default:
				[[likely]]
					buffer.push_back(c);
			}
		if (buffer.size())
		[[unlikely]]
			_dictionary.emplace_back(std::move(buffer)).shrink_to_fit();

		_dictionary.emplace_back(1, ' ');
		_dictionary.shrink_to_fit();
	}

	[[nodiscard]]
	auto forward(
		const std::vector<cv::Mat> & classifier_results,
		size_t batch,
		const cv::Size & shape,
		double threshold
	) &
	{
		std::vector<std::tuple<size_t, std::string, double>> results;
		results.reserve(classifier_results.size());

		int64_t input_shape[] { batch, 3, shape.height, shape.width };
		auto input_tensor = Ort::Value::CreateTensor<float>(_allocator, input_shape, 4);
		int64_t output_shape[] { batch, 40, _dictionary.size() };
		auto output_tensor = Ort::Value::CreateTensor<float>(_allocator, output_shape, 3);
		size_t stride = shape.height * shape.width;

		std::string buffer;
		buffer.reserve(120);
		for (size_t i = 0; i < classifier_results.size(); i += batch)
		{
			size_t left = std::min(batch, classifier_results.size() - i);
			auto input_ptr = input_tensor.GetTensorMutableData<float>();
			for (size_t j = 0, pos = i; j < left; ++j, ++pos)
			{
				cv::Mat transformed;
				transformation::scale_letterbox(classifier_results[pos], transformed, shape, _mean, _stddev);

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
			for (size_t j = 0; j < left; ++j)
			{
				double sum_score = 0;
				size_t count = 0;
				for (size_t k = 0; k < 40; ++k)
				{
					float max_value = std::numeric_limits<float>::min();
					size_t max_index = 0;
					for (size_t l = 0; l < _dictionary.size(); ++l)
						if (auto value = *output_ptr++; value > max_value)
						{
							max_value = value;
							max_index = l;
						}
					if (max_index)
					{
						const auto & selection = _dictionary[max_index];
						buffer.append(selection.data(), selection.size());
						sum_score += max_value;
						++count;
					}
				}
				if (count)
				{
					if (float score = sum_score / count; score >= threshold)
						results.emplace_back(i + j, buffer, score);
					buffer.clear();
				}
			}
		}

		return results;
	}

	void warmup(size_t batch, const cv::Size & shape) &
	{
		int64_t input_shape[] { batch, 3, shape.height, shape.width };
		auto input_tensor = Ort::Value::CreateTensor<float>(_allocator, input_shape, 4);
		int64_t output_shape[] { batch, 40, _dictionary.size() };
		auto output_tensor = Ort::Value::CreateTensor<float>(_allocator, output_shape, 3);
		_model(input_tensor, output_tensor);
	}
};

}

#endif
