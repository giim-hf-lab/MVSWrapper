#include <cstddef>

#include <string>
#include <vector>

#include <mio/mmap.hpp>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "framework/onnxruntime/ocr/recogniser.hpp"

#include "./utils.hpp"

namespace inferences::framework::onnxruntime::ocr
{

recogniser::parameters::parameters(
	size_t batch_size,
	const cv::Size& shape,
	const cv::Scalar& mean,
	const cv::Scalar& stddev,
	double threshold
) noexcept : batch_size(batch_size), shape(shape), mean(mean), stddev(stddev), threshold(threshold) {}

recogniser::parameters::~parameters() noexcept = default;

recogniser::parameters::parameters(const parameters&) noexcept = default;

recogniser::parameters::parameters(parameters&&) noexcept = default;

recogniser::parameters& recogniser::parameters::operator=(const parameters&) noexcept = default;

recogniser::parameters& recogniser::parameters::operator=(parameters&&) noexcept = default;

recogniser::recogniser(
	const std::string& model_path,
	const std::string& dictionary_path,
	const Ort::SessionOptions& options,
	GraphOptimizationLevel graph_opt_level
) : _model(model_path, options, graph_opt_level), _dictionary()
{
	mio::mmap_source dict(dictionary_path);
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

recogniser::recogniser(
	const std::string& model_path,
	const std::string& dictionary_path,
	GraphOptimizationLevel graph_opt_level
) : recogniser(model_path, dictionary_path, _GLOBAL_DEFAULT_OPTIONS, graph_opt_level) {}

recogniser::~recogniser() noexcept = default;

recogniser::recogniser(recogniser&&) noexcept = default;

[[nodiscard]]
std::vector<std::tuple<size_t, std::string, double>> recogniser::operator()(
	const std::vector<cv::Mat>& fragments,
	const parameters& parameters
) &
{
	std::vector<std::tuple<size_t, std::string, double>> results;
	results.reserve(fragments.size());

	int64_t input_shape[] { parameters.batch_size, 3, parameters.shape.height, parameters.shape.width };
	auto input_tensor = model::tensor<float>(input_shape, 4);
	int64_t output_shape[] { parameters.batch_size, 40, _dictionary.size() + 1 };
	auto output_tensor = model::tensor<float>(output_shape, 3);

	size_t input_stride = parameters.shape.area() * 3, output_stride = (_dictionary.size() + 1) * 40;
	for (size_t i = 0; i < fragments.size(); i += parameters.batch_size)
	{
		size_t left = std::min(parameters.batch_size, fragments.size() - i);
		auto write_ptr = input_tensor.GetTensorMutableData<float>();
		for (size_t j = 0; j < left; ++j)
			_scale_split_image(
				fragments[i + j],
				parameters.shape,
				parameters.mean,
				parameters.stddev,
				write_ptr + j * input_stride
			);

		_model(input_tensor, output_tensor);

		auto read_ptr = output_tensor.GetTensorData<float>();
		for (size_t j = 0; j < left; ++j)
		{
			std::string buffer;
			// each UTF-8 character may consume up to 4 bytes
			buffer.reserve(160);

			auto current_read_ptr = read_ptr + j * output_stride;

			double sum_score = 0;
			size_t count = 0;
			for (size_t k = 0; k < 40; ++k)
			{
				auto max_value = std::numeric_limits<float>::min();
				size_t max_index = 0;
				for (size_t l = 0; l <= _dictionary.size(); ++l)
					if (auto value = *current_read_ptr++; value > max_value)
					{
						max_value = value;
						max_index = l;
					}
				if (max_index)
				{
					const auto& selection = _dictionary[max_index - 1];
					buffer.append(selection.data(), selection.size());
					sum_score += max_value;
					++count;
				}
			}

			if (count)
				if (float score = sum_score / count; score >= parameters.threshold)
				{
					buffer.shrink_to_fit();
					results.emplace_back(i + j, std::move(buffer), score);
				}
		}
	}

	return results;
}

void recogniser::warmup(const parameters& parameters) &
{
	int64_t input_shape[] { parameters.batch_size, 3, parameters.shape.height, parameters.shape.width };
	auto input_tensor = model::tensor<float>(input_shape, 4);
	int64_t output_shape[] { parameters.batch_size, 40, _dictionary.size() + 1 };
	auto output_tensor = model::tensor<float>(output_shape, 3);
	_model(input_tensor, output_tensor);
}

}
