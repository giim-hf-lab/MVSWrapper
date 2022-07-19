#ifndef __INFERENCE_TORCHSCRIPT_YOLO_V5_HPP__
#define __INFERENCE_TORCHSCRIPT_YOLO_V5_HPP__

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/script.h>

#include "../../../std.hpp"
#include "./augmentation.hpp"

namespace inference::torchscript::yolo
{

class v5 final
{
	mutable torch::jit::Module _model;
	torch::DeviceType _device_type;
	torch::ScalarType _scalar_type;
	cv::Size _accepted_size;
	bool _scale_up;
	cv::Scalar _padded_colour;

	[[nodiscard]]
	inline auto compute(torch::Tensor tensor) const &
	{
		return _model({ std::move(tensor) }).toTuple()->elements()[0].toTensor();
	}
public:
	v5(
		const std::path_like auto & model_path,
		torch::DeviceType device_type,
		torch::ScalarType scalar_type,
		cv::Size accepted_size,
		bool scale_up,
		cv::Scalar padded_colour
	) :
		_model(torch::jit::load(model_path)),
		_device_type(device_type),
		_scalar_type(scalar_type),
		_accepted_size(std::move(accepted_size)),
		_scale_up(scale_up),
		_padded_colour(std::move(padded_colour))
	{
		_model.to(device_type, scalar_type, false);
		_model.eval();

		torch::InferenceMode guard(true);
		_model({ torch::zeros(
			{ 1, 3, _accepted_size.height, _accepted_size.width },
			torch::TensorOptions(device_type).dtype(scalar_type)
		) });
	}

	v5(const v5 &) = delete;
	v5(v5 &&) noexcept = delete;

	v5 & operator=(const v5 &) = delete;
	v5 & operator=(v5 &&) noexcept = delete;

	[[nodiscard]]
	auto infer_image(
		cv::Mat image,
		double score_threshold,
		double iou_threshold,
		const torch::Tensor & inclusions,
		const torch::Tensor & exclusions
	) const &
	{
		torch::InferenceMode guard(true);

		auto size = image.size();
		bool scaled;
		double ratio;
		int64_t left, right, top, bottom;
		if (size != _accepted_size)
		{
			auto [reshaped, _ratio, _left, _right, _top, _bottom] = letterbox(
				image,
				_accepted_size,
				_scale_up,
				_padded_colour
			);
			scaled = true;
			image = std::move(reshaped);
			ratio = _ratio;
			left = _left;
			right = _right;
			top = _top;
			bottom = _bottom;
		}
		else
			scaled = false;

		cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB);
		auto result = compute(
			torch::from_blob(image.data, { _accepted_size.height, _accepted_size.width, 3 }, torch::kByte)
				.permute({ 2, 0, 1 })
				.unsqueeze(0)
				.to(_device_type, _scalar_type, false, false, torch::MemoryFormat::Contiguous)
				.div(255)
		).select(0, 0);
		xywh2xyxy(result.slice(1, 0, 4), size.height, size.width);
		if (scaled)
			scale_coords(result.slice(1, 0, 4), ratio, left, right, top, bottom);

		auto [boxes, scores, classes] = non_max_suppression(
			std::move(result),
			score_threshold,
			iou_threshold,
			inclusions,
			exclusions,
			std::max(size.height, size.width),
			torch::kCPU,
			torch::kFloat32
		);
		return results_to_vector(std::move(boxes), std::move(scores), std::move(classes));
	}

	[[nodiscard]]
	inline auto infer_image(
		cv::Mat image,
		double score_threshold,
		double iou_threshold,
		const std::vector<int64_t> & inclusions,
		const std::vector<int64_t> & exclusions
	) const &
	{
		auto option = torch::TensorOptions(_device_type)
			.dtype(torch::kInt64)
			.memory_format(torch::MemoryFormat::Contiguous);
		return infer_image(
			std::move(image),
			score_threshold,
			iou_threshold,
			torch::tensor(inclusions, option),
			torch::tensor(exclusions, option)
		);
	}
};

}

#endif
