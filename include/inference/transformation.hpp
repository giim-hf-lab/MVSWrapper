#ifndef __INFERENCE_TRANSFORMATION_HPP__
#define __INFERENCE_TRANSFORMATION_HPP__

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <type_traits>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace inference
{

template<typename Tensor>
	requires std::is_object_v<Tensor>
class transformation final
{
	double ratio;
	int64_t left, right, top, bottom;

	inline transformation() noexcept : ratio(0.0) {}

	inline transformation(
		double ratio,
		int64_t left,
		int64_t right,
		int64_t top,
		int64_t bottom
	) noexcept : ratio(ratio), left(left), right(right), top(top), bottom(bottom) {}
public:
	[[nodiscard]]
	static transformation letterbox(
		cv::Mat & image,
		const cv::Size & new_size,
		bool scale_up,
		const cv::Scalar & padded_colour
	)
	{
		auto image_size = image.size();
		if (image_size == new_size)
			return {};

		auto ratio = std::min(double(new_size.width) / image_size.width, double(new_size.height) / image_size.height);
		cv::Mat resized;
		if (ratio < 1.0 || (ratio > 1.0 && scale_up))
		{
			image_size.height *= ratio;
			image_size.width *= ratio;
			image_size.height = std::clamp(image_size.height, 0, new_size.height);
			image_size.width = std::clamp(image_size.width, 0, new_size.width);

			cv::resize(image, resized, image_size, ratio, ratio, cv::InterpolationFlags::INTER_LINEAR);
		}
		else
		{
			ratio = 1.0;
			resized = std::move(image);
		}

		size_t width_pad = new_size.width - image_size.width, height_pad = new_size.height - image_size.height;
		size_t left = width_pad >> 1, right = width_pad - left;
		size_t top = height_pad >> 1, bottom = height_pad - top;

		cv::copyMakeBorder(resized, image, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT, padded_colour);

		return { ratio, left, right, top, bottom };
	}

	~transformation() noexcept = default;

	transformation(const transformation &) noexcept = default;
	transformation(transformation &&) noexcept = default;

	transformation & operator=(const transformation &) & noexcept = default;
	transformation & operator=(transformation &&) & noexcept = default;

	transformation & operator=(const transformation &) && noexcept = delete;
	transformation & operator=(transformation &&) && noexcept = delete;

	void rescale(Tensor & boxes, const cv::Size & size) const;
};

}

#endif
