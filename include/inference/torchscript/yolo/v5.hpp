#ifndef __INFERENCE_TORCHSCRIPT_YOLO_V5_HPP__
#define __INFERENCE_TORCHSCRIPT_YOLO_V5_HPP__

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <concepts>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include "../filter.hpp"
#include "../transformation.hpp"
#include "./nms.hpp"

namespace inference::torchscript::yolo
{

class v5 final
{
	cv::Size _image_size;
	bool _scale_up;
	cv::Scalar _padded_colour;
	torch::DeviceType _device_type;
	torch::ScalarType _scalar_type;
	std::vector<std::string_view> _labels;
	std::unordered_map<std::string, size_t> _labels_indicies;
	torch::jit::Module _model;

	[[nodiscard]]
	[[using gnu : always_inline]]
	inline auto _init_model(const std::string & model_path) &
	{
		torch::InferenceMode guard(true);

		torch::jit::ExtraFilesMap extra_files_map { { "config.txt", "" } };
		auto ret = torch::jit::load(model_path, _device_type, extra_files_map);
		ret.to(_device_type, _scalar_type, false);

		// https://github.com/ultralytics/yolov5/blob/v6.1/export.py#L99-L100
		const auto & config_str = extra_files_map["config.txt"];
		if (config_str.empty())
		[[unlikely]]
			throw std::runtime_error("the model doesn't contain the required metadata");
		auto config = nlohmann::json::parse(config_str);

		auto parsed_labels = config["names"].get<std::vector<std::string>>();
		auto labels_num = parsed_labels.size();
		_labels.reserve(labels_num);
		_labels_indicies.reserve(labels_num);
		size_t index = 0;
		for (auto & label : parsed_labels)
			if (auto [it, emplaced] = _labels_indicies.try_emplace(std::move(label), index); emplaced)
			[[likely]]
			{
				_labels.emplace_back(it->first);
				++index;
			}
			else
			[[unlikely]]
				throw std::runtime_error(fmt::format(FMT_COMPILE("duplicate label <{}>"), it->first));
		_labels.shrink_to_fit();
		_labels_indicies.rehash(index);

		// https://github.com/ultralytics/yolov5/blob/v6.1/export.py#L457
		auto st = config["shape"].get<std::tuple<size_t, size_t, size_t, size_t>>();
		if (const auto & [N, C, H, W] = st; N == 1 && C == 3)
		[[likely]]
			_image_size = { H, W };
		else
		[[unlikely]]
			throw std::runtime_error(fmt::format(FMT_COMPILE("input shape ({}) unrecognised"), fmt::join(st, ", ")));

		ret.eval();
		return ret;
	}

	[[nodiscard]]
	[[using gnu : always_inline]]
	inline auto labels_to_indices(const std::vector<std::string> & labels) const &
	{
		std::vector<int64_t> indicies;
		if (auto size = labels.size(); size)
		{
			indicies.reserve(size);
			for (const auto & label : labels)
				indicies.emplace_back(_labels_indicies.at(label));
			std::ranges::sort(indicies);
			auto trim_ranges = std::ranges::unique(indicies);
			indicies.erase(trim_ranges.begin(), trim_ranges.end());
		}
		return indicies;
	}
public:
	v5(
		const std::string & model_path,
		torch::DeviceType device_type,
		torch::ScalarType scalar_type,
		bool scale_up,
		// https://github.com/ultralytics/yolov5/blob/v6.1/utils/augmentations.py#L91
		cv::Scalar padded_colour = { 114, 114, 114 }
	) :
		_image_size(),
		_scale_up(scale_up),
		_padded_colour(std::move(padded_colour)),
		_device_type(device_type),
		_scalar_type(scalar_type),
		_labels(),
		_labels_indicies(),
		_model(_init_model(model_path)) {}

	v5(const v5 &) = delete;
	v5(v5 &&) noexcept = default;

	v5 & operator=(const v5 &) = delete;
	v5 & operator=(v5 &&) = delete;

	[[nodiscard]]
	filter create_filter(const std::vector<std::string> & inclusion, const std::vector<std::string> & exclusion) const &
	{
		auto option = torch::TensorOptions(_device_type)
			.dtype(torch::kInt64)
			.memory_format(torch::MemoryFormat::Contiguous);
		return {
			torch::tensor(labels_to_indices(inclusion), option),
			torch::tensor(labels_to_indices(exclusion), option)
		};
	}

	[[nodiscard]]
	auto forward(cv::Mat image, double score_threshold, double iou_threshold, const filter & label_filter) &
	{
		auto original_size = image.size();
		auto scaler = transformation::letterbox(image, image, _image_size, _scale_up, _padded_colour);
		cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB);

		torch::InferenceMode guard(true);

		auto result = _model({
			torch::from_blob(image.data, { _image_size.height, _image_size.width, 3 }, torch::kByte)
				// https://github.com/ultralytics/yolov5/blob/v6.1/export.py#L509
				.permute({ 2, 0, 1 })
				.unsqueeze(0)
				.to(_device_type, _scalar_type, false, false, torch::MemoryFormat::Contiguous)
				.div(255)
		}).toTuple()->elements()[0].toTensor();
		xywh2xyxy(result, _image_size);
		scaler.rescale(result, original_size);

		auto [results_left, boxes, scores, classes] = non_max_suppression(
			result.select(0, 0),
			score_threshold,
			iou_threshold,
			label_filter,
			std::max(original_size.height, original_size.width),
			torch::kCPU,
			torch::kFloat32
		);
		const auto * boxes_ptr = boxes.data_ptr<int64_t>();
		const auto * scores_ptr = scores.data_ptr<float>();
		const auto * classes_ptr = classes.data_ptr<int64_t>();
		std::vector<std::tuple<std::string, float, cv::Point, cv::Point>> ret;
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

	void warmup() &
	{
		torch::InferenceMode guard(true);
		_model({ torch::zeros(
			{ 1, 3, _image_size.height, _image_size.width },
			torch::TensorOptions(_device_type).dtype(_scalar_type)
		) });
	}
};

}

#endif
