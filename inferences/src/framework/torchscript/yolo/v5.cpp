#include <cstddef>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <fmt/compile.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <torch/script.h>

#include "framework/torchscript/filter.hpp"
#include "framework/torchscript/yolo/v5.hpp"

#include "../../transformation.hpp"
#include "../transformation.hpp"
#include "./nms.hpp"

namespace
{

[[nodiscard]]
inline static torch::jit::Module _init_model(
	const std::string& model_path,
	torch::DeviceType device_type,
	torch::ScalarType scalar_type,
	std::vector<std::string_view>& labels,
	std::unordered_map<std::string, size_t>& labels_indicies,
	cv::Size& image_size
)
{
	torch::InferenceMode guard(true);

	torch::jit::ExtraFilesMap extra_files_map { { "config.txt", "" } };
	auto ret = torch::jit::load(model_path, device_type, extra_files_map);
	ret.to(device_type, scalar_type, false);

	// https://github.com/ultralytics/yolov5/blob/v6.1/export.py#L99-L100
	const auto& config_str = extra_files_map["config.txt"];
	if (config_str.empty())
	[[unlikely]]
		throw std::runtime_error("the model doesn't contain the required metadata");
	auto config = nlohmann::json::parse(config_str);

	auto parsed_labels = config["names"].get<std::vector<std::string>>();
	auto labels_num = parsed_labels.size();
	labels.reserve(labels_num);
	labels_indicies.reserve(labels_num);
	size_t index = 0;
	for (auto& label : parsed_labels)
		if (auto [it, emplaced] = labels_indicies.try_emplace(std::move(label), index); emplaced)
		[[likely]]
		{
			labels.emplace_back(it->first);
			++index;
		}
		else
		[[unlikely]]
			throw std::runtime_error(fmt::format(FMT_COMPILE("duplicate label <{}>"), it->first));
	labels.shrink_to_fit();
	labels_indicies.rehash(index);

	// https://github.com/ultralytics/yolov5/blob/v6.1/export.py#L457
	auto st = config["shape"].get<std::tuple<size_t, size_t, size_t, size_t>>();
	if (const auto& [N, C, H, W] = st; N == 1 && C == 3)
	[[likely]]
		image_size = { H, W };
	else
	[[unlikely]]
		throw std::runtime_error(fmt::format(FMT_COMPILE("input shape ({}) unrecognised"), fmt::join(st, ", ")));

	ret.eval();
	return ret;
}

[[nodiscard]]
inline static std::vector<int64_t> _labels_to_indicies(
	const std::unordered_map<std::string, size_t>& labels_indicies,
	const std::vector<std::string>& labels
)
{
	std::vector<int64_t> result;
	if (auto size = labels.size())
	{
		result.reserve(size);
		for (const auto& label : labels)
			result.emplace_back(labels_indicies.at(label));
		std::ranges::sort(result);
		auto trim_ranges = std::ranges::unique(result);
		result.erase(trim_ranges.begin(), trim_ranges.end());
	}
	return result;
}

}

namespace inferences::framework::torchscript::yolo
{

v5::parameters::parameters(bool scale_up, double score_threshold, double iou_threshold) noexcept :
	scale_up(scale_up),
	score_threshold(score_threshold),
	iou_threshold(iou_threshold) {}

v5::parameters::~parameters() noexcept = default;

v5::parameters::parameters(const parameters& other) noexcept = default;

v5::parameters::parameters(parameters&&) noexcept = default;

v5::parameters& v5::parameters::operator=(const parameters& other) noexcept = default;

v5::parameters& v5::parameters::operator=(parameters&&) noexcept = default;

v5::v5(
	const std::string& model_path,
	torch::DeviceType device_type,
	torch::ScalarType scalar_type
) :
	_image_size(),
	_device_type(device_type),
	_scalar_type(scalar_type),
	_labels(),
	_labels_indicies(),
	_model(_init_model(model_path, device_type, scalar_type, _labels, _labels_indicies, _image_size)) {}

v5::~v5() noexcept = default;

v5::v5(v5&&) noexcept = default;

[[nodiscard]]
std::vector<std::tuple<std::string, float, cv::Point, cv::Point>> v5::operator()(
	const cv::Mat& image,
	const parameters& parameters,
		const filter& label_filter
) &
{
	// https://github.com/ultralytics/yolov5/blob/v6.1/utils/augmentations.py#L91
	static const cv::Scalar PADDING_FILL = CV_RGB(114, 114, 114);

	auto original_size = image.size();
	cv::Mat transformed;
	auto scaler = transformation::letterbox(image, transformed, _image_size, parameters.scale_up, PADDING_FILL);
	cv::cvtColor(transformed.empty() ? image : transformed, transformed, cv::ColorConversionCodes::COLOR_BGR2RGB);

	torch::InferenceMode guard(true);

	auto computed = _model({
		torch::from_blob(transformed.data, { _image_size.height, _image_size.width, 3 }, torch::kByte)
			// https://github.com/ultralytics/yolov5/blob/v6.1/export.py#L509
			.permute({ 2, 0, 1 })
			.unsqueeze(0)
			.to(_device_type, _scalar_type, false, false, torch::MemoryFormat::Contiguous) / 255
	}).toTuple()->elements()[0].toTensor();
	xywh2xyxy(computed, _image_size);
	scaler.rescale(computed, original_size);

	auto [results_left, boxes, scores, classes] = non_max_suppression(
		computed.select(0, 0),
		parameters.score_threshold,
		parameters.iou_threshold,
		label_filter,
		std::max(original_size.height, original_size.width),
		torch::kCPU,
		torch::kFloat32
	);
	const auto *boxes_ptr = boxes.data_ptr<int64_t>();
	const auto *scores_ptr = scores.data_ptr<float>();
	const auto *classes_ptr = classes.data_ptr<int64_t>();
	std::vector<std::tuple<std::string, float, cv::Point, cv::Point>> result;
	result.reserve(results_left);
	for (size_t i = 0; i < results_left; ++i)
	{
		auto offset_boxes_ptr = boxes_ptr + i * 4;
		result.emplace_back(
			_labels[classes_ptr[i]],
			scores_ptr[i],
			cv::Point { offset_boxes_ptr[0], offset_boxes_ptr[1] },
			cv::Point { offset_boxes_ptr[2], offset_boxes_ptr[3] }
		);
	}
	return result;
}

[[nodiscard]]
filter v5::create_filter(const std::vector<std::string>& inclusion, const std::vector<std::string>& exclusion) const&
{
	auto option = torch::TensorOptions(_device_type)
		.dtype(torch::kInt64)
		.memory_format(torch::MemoryFormat::Contiguous);
	return {
		torch::tensor(_labels_to_indicies(_labels_indicies, inclusion), option),
		torch::tensor(_labels_to_indicies(_labels_indicies, exclusion), option)
	};
}

void v5::warmup() &
{
	torch::InferenceMode guard(true);
	_model({ torch::zeros(
		{ 1, 3, _image_size.height, _image_size.width },
		torch::TensorOptions(_device_type).dtype(_scalar_type)
	) });
}

}
