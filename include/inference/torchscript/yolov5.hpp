#ifndef __INFERENCE_TORCHSCRIPT_YOLOV5_HPP__
#define __INFERENCE_TORCHSCRIPT_YOLOV5_HPP__

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <fmt/ranges.h>
#include <mio/mmap.hpp>
#include <opencv2/core.hpp>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <torch/script.h>
#include <torchvision/ops/nms.h>

#include "../../std.hpp"

namespace inference::torchscript::yolov5
{

class engine final
{
	torch::jit::Module _model;
	std::vector<std::string> _labels;
	std::unordered_map<std::string_view, size_t> _labels_mapping;

	auto non_max_suppression(
		torch::Tensor result,
		double score_threshold,
		double iou_threshold
	) const
	{
		std::vector<std::tuple<std::string, double, cv::Point, cv::Point>> ret;

		score_threshold = std::clamp(score_threshold, 0.0, 1.0);
		iou_threshold = std::clamp(iou_threshold, 0.0, 1.0);

		auto indices = (result.select(1, 4) >= score_threshold).nonzero().squeeze(1);
		if (!indices.size(0))
			return ret;

		result = result.index_select(0, indices);
		auto [scores, classes] = (result.slice(1, 5) * result.slice(1, 4, 5)).max(1, true);
		scores = scores.squeeze(1);
		indices = (scores >= score_threshold).nonzero().squeeze(1);
		size_t results_left = indices.size(0);
		if (!results_left)
			return ret;

		scores = scores.index_select(0, indices);
		classes = classes.index_select(0, indices);

		auto boxes = result.slice(1, 0, 4).index_select(0, indices);
		auto sx = boxes.select(1, 2) / 2, sy = boxes.select(1, 3) / 2;
		auto cx = boxes.select(1, 0), cy = boxes.select(1, 1);
		boxes.select(1, 2) = (cx + sx).clamp(0, 640);
		boxes.select(1, 3) = (cy + sy).clamp(0, 640);
		boxes.select(1, 0) = (cx - sx).clamp(0, 640);
		boxes.select(1, 1) = (cy - sy).clamp(0, 640);
		indices = vision::ops::nms(boxes + classes * 640, scores, iou_threshold);
		results_left = indices.size(0);
		if (!results_left)
			return ret;

		boxes = boxes.index_select(0, indices).to(torch::kCPU, torch::kInt64).contiguous();
		scores = scores.index_select(0, indices).to(torch::kCPU, torch::kFloat64).contiguous();
		classes = classes.squeeze(1).index_select(0, indices).to(torch::kCPU, torch::kInt64).contiguous();

		const auto * boxes_ptr = boxes.data<int64_t>();
		const auto * scores_ptr = scores.data<double>();
		const auto * classes_ptr = classes.data<int64_t>();
		ret.reserve(results_left);
		for (size_t i = 0; i < results_left; ++i)
		{
			auto offset_boxes_ptr = boxes_ptr + i * 4;
			ret.emplace_back(
				_labels[classes_ptr[i]],
				scores_ptr[i],
				cv::Point { offset_boxes_ptr[0], offset_boxes_ptr[1] },
				cv::Point { offset_boxes_ptr[2], offset_boxes_ptr[3] }
			);
		}

		return ret;
	}
public:
	engine(const std::path_like auto & model_path, const std::path_like auto & labels_path) :
		_model(torch::jit::load(model_path)),
		_labels(),
		_labels_mapping()
	{
		_model.eval();
		_model.to(torch::kCUDA, torch::kFloat16);

		torch::InferenceMode guard(true);
		_model({ torch::zeros({ 1, 3, 640, 640 }, torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat16)) });

		auto labels_mmap = mio::mmap_source(labels_path);
		std::string buffer(labels_mmap.data(), labels_mmap.size());
		for (size_t pos = 0, new_pos = 0, i = 0; new_pos < buffer.size() && pos < buffer.size(); pos = new_pos + 1)
		{
			new_pos = std::min(buffer.size(), buffer.find('\n', pos));
			_labels_mapping.emplace(_labels.emplace_back(buffer.c_str() + pos, new_pos - pos), i++);
		}
		_labels.shrink_to_fit();
	}

	engine(const engine &) = delete;
	engine(engine &&) noexcept = delete;

	engine & operator=(const engine &) = delete;
	engine & operator=(engine &&) noexcept = delete;

	[[nodiscard]]
	auto operator()(torch::Tensor tensor, double score_threshold, double iou_threshold)
	{
		torch::InferenceMode guard(true);

		return non_max_suppression(
			_model({ tensor }).toTuple()->elements()[0].toTensor().select(0, 0),
			score_threshold,
			iou_threshold
		);
	}

	[[nodiscard]]
	auto infer_image(cv::Mat image, double score_threshold, double iou_threshold)
	{
		auto size = image.size();
		return operator()(
			torch::from_blob(image.data, { size.height, size.width, 3 }, torch::kByte)
				.permute({ 2, 0, 1 })
				.unsqueeze(0)
				.cuda()
				.toType(torch::kFloat16)
				.div(255),
			score_threshold,
			iou_threshold
		);
	}
};

}

#endif
