#include <cstddef>

#include <string>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "framework/onnxruntime/ocr/classifier.hpp"

#include "./utils.hpp"

namespace
{

inline static void _crop_box(const cv::Mat& image, const cv::RotatedRect& box, cv::Mat& result)
{
	const auto& size = box.size;
	cv::Point2f vertices[4], upright_vertices[4] {
		{ 0, size.height },
		{ 0, 0 },
		{ size.width, 0 },
		{ size.width, size.height }
	};
	box.points(vertices);
	cv::warpPerspective(
		image,
		result,
		cv::getPerspectiveTransform(vertices, upright_vertices),
		size
	);
	if (size.height > size.width)
		cv::rotate(result, result, cv::RotateFlags::ROTATE_90_CLOCKWISE);
}

}

namespace inferences::framework::onnxruntime::ocr
{

classifier::parameters::parameters(
	size_t batch_size,
	const cv::Size& shape,
	const cv::Scalar& mean,
	const cv::Scalar& stddev,
	double threshold
) noexcept : batch_size(batch_size), shape(shape), mean(mean), stddev(stddev), threshold(threshold) {}

classifier::parameters::~parameters() noexcept = default;

classifier::parameters::parameters(const parameters&) noexcept = default;

classifier::parameters::parameters(parameters&&) noexcept = default;

classifier::parameters& classifier::parameters::operator=(const parameters&) noexcept = default;

classifier::parameters& classifier::parameters::operator=(parameters&&) noexcept = default;

classifier::classifier(
	const std::string& model_path,
	const Ort::SessionOptions& options,
	GraphOptimizationLevel graph_opt_level
) : _model(model_path, options, graph_opt_level) {}

classifier::classifier(
	const std::string& model_path,
	GraphOptimizationLevel graph_opt_level
) : classifier(model_path, _GLOBAL_DEFAULT_OPTIONS, graph_opt_level) {}

classifier::~classifier() noexcept = default;

classifier::classifier(classifier&&) noexcept = default;

[[nodiscard]]
std::vector<cv::Mat> classifier::operator()(
	const cv::Mat& image,
	const std::vector<cv::RotatedRect>& boxes,
	const parameters& parameters
) &
{
	std::vector<cv::Mat> results(boxes.size());
	for (size_t i = 0; i < boxes.size(); ++i)
		_crop_box(image, boxes[i], results[i]);

	int64_t input_shape[] { parameters.batch_size, 3, parameters.shape.height, parameters.shape.width };
	auto input_tensor = model::tensor<float>(input_shape, 4);
	int64_t output_shape[] { parameters.batch_size, 2 };
	auto output_tensor = model::tensor<float>(output_shape, 2);

	size_t stride = parameters.shape.area() * 3;
	for (size_t i = 0; i < results.size(); i += parameters.batch_size)
	{
		size_t left = std::min(parameters.batch_size, results.size() - i);
		auto write_ptr = input_tensor.GetTensorMutableData<float>();
		for (size_t j = 0; j < left; ++j)
			_scale_split_image(
				results[i + j],
				parameters.shape,
				parameters.mean,
				parameters.stddev,
				write_ptr + j * stride
			);

		_model(input_tensor, output_tensor);

		auto read_ptr = output_tensor.GetTensorData<float>();
		for (size_t j = 0; j < left; ++j)
		{
			size_t pos = 2 * j;
			if (auto s1 = read_ptr[pos], s2 = read_ptr[pos + 1]; s1 < s2 and s2 > parameters.threshold)
			{
				auto& result = results[i + j];
				cv::rotate(result, result, cv::RotateFlags::ROTATE_180);
			}
		}
	}

	return results;
}

void classifier::warmup(const parameters& parameters) &
{
	int64_t input_shape[] { parameters.batch_size, 3, parameters.shape.height, parameters.shape.width };
	auto input_tensor = model::tensor<float>(input_shape, 4);
	int64_t output_shape[] { parameters.batch_size, 2 };
	auto output_tensor = model::tensor<float>(output_shape, 2);
	_model(input_tensor, output_tensor);
}

}
