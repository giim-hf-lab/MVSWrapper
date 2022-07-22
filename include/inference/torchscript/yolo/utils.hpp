#ifndef __INFERENCE_TORCHSCRIPT_YOLO_UTILS_HPP__
#define __INFERENCE_TORCHSCRIPT_YOLO_UTILS_HPP__

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/script.h>
#include <torchvision/ops/nms.h>

namespace inference::torchscript::yolo
{

[[nodiscard]]
inline static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> empty_result(
	torch::DeviceType device_type,
	torch::ScalarType scores_scalar_type
)
{
	auto option = torch::TensorOptions(device_type).memory_format(torch::MemoryFormat::Contiguous);
	return {
		torch::empty({ 0, 4 }, option.dtype(torch::kInt64)),
		torch::tensor({}, option.dtype(scores_scalar_type)),
		torch::tensor({}, option.dtype(torch::kInt64))
	};
}

[[nodiscard]]
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> non_max_suppression(
	torch::Tensor result,
	double score_threshold,
	double iou_threshold,
	const torch::Tensor & inclusions,
	const torch::Tensor & exclusions,
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

	if (inclusions.size(0))
	{
		indices = (classes == inclusions).any(1, false).nonzero().squeeze(1);
		if (!indices.size(0))
			return empty_result(results_device_type, results_scores_scalar_type);

		boxes = boxes.index_select(0, indices);
		scores = scores.index_select(0, indices);
		classes = classes.index_select(0, indices);
	}

	if (exclusions.size(0))
	{
		indices = (classes != exclusions).all(1, false).nonzero().squeeze(1);
		if (!indices.size(0))
			return empty_result(results_device_type, results_scores_scalar_type);

		boxes = boxes.index_select(0, indices);
		scores = scores.index_select(0, indices);
		classes = classes.index_select(0, indices);
	}

	indices = vision::ops::nms(boxes + classes * max_wh, scores, iou_threshold);
	auto results_left = indices.size(0);
	if (!results_left)
		return empty_result(results_device_type, results_scores_scalar_type);

	boxes = boxes
		.index_select(0, indices)
		.to(results_device_type, torch::kInt64, false, false, torch::MemoryFormat::Contiguous);
	scores = scores
		.index_select(0, indices)
		.to(results_device_type, torch::kFloat32, false, false, torch::MemoryFormat::Contiguous);
	classes = classes
		.squeeze(1)
		.index_select(0, indices)
		.to(results_device_type, torch::kInt64, false, false, torch::MemoryFormat::Contiguous);

	return { std::move(boxes), std::move(scores), std::move(classes) };
}

[[nodiscard]]
static std::vector<std::tuple<size_t, float, cv::Point, cv::Point>> results_to_vector(
	torch::Tensor boxes,
	torch::Tensor scores,
	torch::Tensor classes
)
{
	auto results_left = boxes.size(0);
	const auto * boxes_ptr = boxes.data<int64_t>();
	const auto * scores_ptr = scores.data<float>();
	const auto * classes_ptr = classes.data<int64_t>();

	std::vector<std::tuple<size_t, float, cv::Point, cv::Point>> ret;
	ret.reserve(results_left);
	for (size_t i = 0; i < results_left; ++i)
	{
		auto offset_boxes_ptr = boxes_ptr + i * 4;
		ret.emplace_back(
			classes_ptr[i],
			scores_ptr[i],
			cv::Point { offset_boxes_ptr[0], offset_boxes_ptr[1] },
			cv::Point { offset_boxes_ptr[2], offset_boxes_ptr[3] }
		);
	}
	return ret;
}

}

#endif
