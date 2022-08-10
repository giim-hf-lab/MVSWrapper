#ifndef __INFERENCE_TORCHSCRIPT_YOLO_NMS_HPP__
#define __INFERENCE_TORCHSCRIPT_YOLO_NMS_HPP__

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <tuple>

#include <torch/script.h>
#include <torchvision/ops/nms.h>

#include "../filter.hpp"

namespace inference::torchscript::yolo
{

[[nodiscard]]
[[using gnu : always_inline]]
inline static std::tuple<size_t, torch::Tensor, torch::Tensor, torch::Tensor> empty_result(
	torch::DeviceType device_type,
	torch::ScalarType scores_scalar_type
)
{
	auto option = torch::TensorOptions(device_type).memory_format(torch::MemoryFormat::Contiguous);
	return {
		0,
		torch::empty({ 0, 4 }, option.dtype(torch::kInt64)),
		torch::empty(0, option.dtype(scores_scalar_type)),
		torch::empty(0, option.dtype(torch::kInt64))
	};
}

[[nodiscard]]
static std::tuple<size_t, torch::Tensor, torch::Tensor, torch::Tensor> non_max_suppression(
	torch::Tensor result,
	double score_threshold,
	double iou_threshold,
	const filter & label_filter,
	int64_t max_wh,
	torch::DeviceType results_device_type,
	torch::ScalarType results_scores_scalar_type
)
{
	score_threshold = std::clamp(score_threshold, 0.0, 1.0);
	iou_threshold = std::clamp(iou_threshold, 0.0, 1.0);
	max_wh = std::max(int64_t(0), max_wh);

	auto indices = (result.select(1, 4) >= score_threshold).nonzero().squeeze(1);
	if (!indices.size(0))
		return empty_result(results_device_type, results_scores_scalar_type);

	result = result.index_select(0, indices);
	auto [scores, classes] = (result.slice(1, 5) * result.slice(1, 4, 5)).max(1, true);
	scores = scores.squeeze(1);
	indices = (scores >= score_threshold).nonzero().squeeze(1);
	if (!indices.size(0))
		return empty_result(results_device_type, results_scores_scalar_type);

	auto boxes = result.index({ indices, torch::indexing::Slice(0, 4) });
	scores = scores.index_select(0, indices);
	classes = classes.index_select(0, indices);

	if (const auto & inclusion = label_filter.inclusion; inclusion.size(0))
	{
		indices = (classes == inclusion).any(1, false).nonzero().squeeze(1);
		if (!indices.size(0))
			return empty_result(results_device_type, results_scores_scalar_type);

		boxes = boxes.index_select(0, indices);
		scores = scores.index_select(0, indices);
		classes = classes.index_select(0, indices);
	}

	if (const auto & exclusion = label_filter.exclusion; exclusion.size(0))
	{
		indices = (classes != exclusion).all(1, false).nonzero().squeeze(1);
		if (!indices.size(0))
			return empty_result(results_device_type, results_scores_scalar_type);

		boxes = boxes.index_select(0, indices);
		scores = scores.index_select(0, indices);
		classes = classes.index_select(0, indices);
	}

	indices = vision::ops::nms(boxes + classes * max_wh, scores, iou_threshold);
	if (auto results_left = indices.size(0); results_left)
		return {
			results_left,
			boxes
				.index_select(0, indices)
				.to(results_device_type, torch::kInt64, false, false, torch::MemoryFormat::Contiguous),
			scores
				.index_select(0, indices)
				.to(results_device_type, results_scores_scalar_type, false, false, torch::MemoryFormat::Contiguous),
			classes
				.squeeze(1)
				.index_select(0, indices)
				.to(results_device_type, torch::kInt64, false, false, torch::MemoryFormat::Contiguous)
		};
	return empty_result(results_device_type, results_scores_scalar_type);
}

}

#endif
