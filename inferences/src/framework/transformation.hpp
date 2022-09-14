#ifndef _INFERENCES_ENGINES_INTERNALS__INFERENCE_TRANSFORMATION_HPP_
#define _INFERENCES_ENGINES_INTERNALS__INFERENCE_TRANSFORMATION_HPP_

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace inferences
{

namespace
{

[[nodiscard]]
inline static double _scale_image(
	const cv::Mat& src,
	cv::Size& src_size,
	cv::Mat& dest,
	const cv::Size& dest_size,
	bool scale_up,
	cv::InterpolationFlags interpolation
)
{
	if (
		auto ratio = std::min(double(dest_size.width) / src_size.width, double(dest_size.height) / src_size.height);
		ratio < 1.0 or (scale_up and ratio > 1.0)
	)
	{
		src_size.height *= ratio;
		src_size.width *= ratio;
		src_size.height = std::clamp(src_size.height, 0, dest_size.height);
		src_size.width = std::clamp(src_size.width, 0, dest_size.width);

		cv::resize(src, dest, src_size, ratio, ratio, interpolation);
		return ratio;
	}
	return 1.0;
}

[[nodiscard]]
inline static std::tuple<int64_t, int64_t, int64_t, int64_t> _pad_image(
	const cv::Mat& src,
	const cv::Size& src_size,
	cv::Mat& dest,
	const cv::Size& dest_size,
	const cv::Scalar& padding
)
{

	if (
		int64_t height_pad = dest_size.height - src_size.height, width_pad = dest_size.width - src_size.width;
		height_pad or width_pad
	)
	{
		int64_t top = height_pad >> 1, bottom = height_pad - top;
		int64_t left = width_pad >> 1, right = width_pad - left;
		cv::copyMakeBorder(src, dest, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT, padding);
		return { left, right, top, bottom };
	}
	return { 0, 0, 0, 0 };
}

struct transformation final
{
	static transformation letterbox(
		const cv::Mat& src,
		cv::Mat& dest,
		const cv::Size& dest_size,
		bool scale_up,
		const cv::Scalar& padding,
		cv::InterpolationFlags interpolation = cv::InterpolationFlags::INTER_LINEAR
	)
	{
		auto src_size = src.size();
		auto ratio = _scale_image(src, src_size, dest, dest_size, scale_up, interpolation);
		auto [left, right, top, bottom] = _pad_image(dest.empty() ? src : dest, src_size, dest, dest_size, padding);
		return { ratio, left, right, top, bottom };
	}

	static transformation letterbox(
		const cv::Mat& src,
		cv::Mat& dest,
		const cv::Size& dest_size,
		bool scale_up,
		const cv::Scalar& mean,
		const cv::Scalar& stddev,
		cv::InterpolationFlags interpolation = cv::InterpolationFlags::INTER_LINEAR
	)
	{
		auto src_size = src.size();
		auto ratio = _scale_image(src, src_size, dest, dest_size, scale_up, interpolation);
		(dest.empty() ? src : dest).convertTo(dest, CV_32FC3, 1.0 / 255.0, 0.0);
		dest -= mean;
		dest /= stddev;
		static const auto BLANK_FILLING = CV_RGB(0, 0, 0);
		auto [left, right, top, bottom] = _pad_image(dest, src_size, dest, dest_size, BLANK_FILLING);
		return { ratio, left, right, top, bottom };
	}

	double ratio;
	int64_t left, right, top, bottom;

	~transformation() noexcept = default;

	transformation(const transformation&) noexcept = default;

	transformation(transformation&&) noexcept = default;

	transformation& operator=(const transformation&) noexcept = default;

	transformation& operator=(transformation&&) noexcept = default;

	template<typename Tensor>
		requires std::is_object_v<Tensor>
	void rescale(Tensor& boxes, const cv::Size& size) const;
private:
	inline transformation(
		double ratio = 1.0,
		int64_t left = 0,
		int64_t right = 0,
		int64_t top = 0,
		int64_t bottom = 0
	) noexcept : ratio(ratio), left(left), right(right), top(top), bottom(bottom) {}
};

template<>
void transformation::rescale<std::vector<cv::RotatedRect>>(
	std::vector<cv::RotatedRect>& boxes,
	const cv::Size& size
) const
{
	for (size_t i = 0; i < boxes.size(); ++i)
	{
		auto& box = boxes[i];
		auto& centre = box.center;
		centre.x = std::clamp((centre.x - left) / ratio, 0.0, double(size.width));
		centre.y = std::clamp((centre.y - top) / ratio, 0.0, double(size.height));
		auto& size = box.size;
		size.width /= ratio;
		size.height /= ratio;
	}
}

}

}

#endif
