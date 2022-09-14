#ifndef INFERENCES_FRAMEWORK_TORCHSCRIPT_YOLO_V5_HPP
#define INFERENCES_FRAMEWORK_TORCHSCRIPT_YOLO_V5_HPP

#include <cstddef>

#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include "../filter.hpp"

namespace inferences::framework::torchscript::yolo
{

struct v5 final
{
	struct parameters final
	{
		bool scale_up;
		double score_threshold, iou_threshold;

		parameters(bool scale_up, double score_threshold, double iou_threshold) noexcept;

		~parameters() noexcept;

		parameters(const parameters&) noexcept;

		parameters(parameters&&) noexcept;

		parameters& operator=(const parameters&) noexcept;

		parameters& operator=(parameters&&) noexcept;
	};
private:
	cv::Size _image_size;
	torch::DeviceType _device_type;
	torch::ScalarType _scalar_type;
	std::vector<std::string_view> _labels;
	std::unordered_map<std::string, size_t> _labels_indicies;
	torch::jit::Module _model;
public:
	v5(const std::string& model_path, torch::DeviceType device_type, torch::ScalarType scalar_type);

	~v5() noexcept;

	v5(const v5&) = delete;

	v5(v5&&) noexcept;

	v5& operator=(const v5&) = delete;

	v5& operator=(v5&&) = delete;

	[[nodiscard]]
	std::vector<std::tuple<std::string, float, cv::Point, cv::Point>> operator()(
		const cv::Mat& image,
		const parameters& parameters,
		const filter& label_filter
	) &;

	[[nodiscard]]
	filter create_filter(const std::vector<std::string>& inclusion, const std::vector<std::string>& exclusion) const&;

	void warmup() &;
};

}

#endif
