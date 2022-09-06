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

struct transformation final
{
	const double ratio;
	const int64_t left, right, top, bottom;

	static transformation letterbox(
		const cv::Mat & src,
		cv::Mat & dest,
		const cv::Size & dest_size,
		bool scale_up,
		const cv::Scalar & padded_colour
	)
	{
		auto src_size = src.size();
		if (src_size == dest_size)
			return {};

		auto ratio = std::min(double(dest_size.width) / src_size.width, double(dest_size.height) / src_size.height);
		if (ratio < 1.0 or (ratio > 1.0 and scale_up))
		{
			src_size.height *= ratio;
			src_size.width *= ratio;
			src_size.height = std::clamp(src_size.height, 0, dest_size.height);
			src_size.width = std::clamp(src_size.width, 0, dest_size.width);

			cv::resize(src, dest, src_size);
		}
		else
			ratio = 1.0;

		size_t width_pad = dest_size.width - src_size.width, height_pad = dest_size.height - src_size.height;
		size_t left = width_pad >> 1, right = width_pad - left;
		size_t top = height_pad >> 1, bottom = height_pad - top;

		cv::copyMakeBorder(dest, dest, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT, padded_colour);

		return { ratio, left, right, top, bottom };
	}

	static transformation scale_zoom(
		const cv::Mat & src,
		cv::Mat & dest,
		size_t side_length,
		bool side_length_as_max,
		const cv::Scalar & mean,
		const cv::Scalar & stddev
	)
	{
		auto src_size = src.size();
		double ratio;
		if ((src_size.height > src_size.width) == side_length_as_max)
		{
			ratio = double(side_length) / src_size.height;
			src_size.height = side_length;
			src_size.width *= ratio;
		}
		else
		{
			ratio = double(side_length) / src_size.width;
			src_size.width = side_length;
			src_size.height *= ratio;
		}
		cv::resize(src, dest, src_size);

		dest.convertTo(dest, CV_32FC3, 1.0 / 255.0, 0.0);
		dest -= mean;
		dest /= stddev;

		return { ratio };
	}

	static transformation scale_letterbox(
		const cv::Mat & src,
		cv::Mat & dest,
		const cv::Size & dest_size,
		const cv::Scalar & mean,
		const cv::Scalar & stddev
	)
	{
		auto src_size = src.size();
		if (src_size == dest_size)
			return {};

		auto ratio = std::min(double(dest_size.width) / src_size.width, double(dest_size.height) / src_size.height);
		src_size.height *= ratio;
		src_size.width *= ratio;
		src_size.height = std::clamp(src_size.height, 0, dest_size.height);
		src_size.width = std::clamp(src_size.width, 0, dest_size.width);
		cv::resize(src, dest, src_size);

		dest.convertTo(dest, CV_32FC3, 1.0 / 255.0, 0.0);
		dest -= mean;
		dest /= stddev;

		size_t width_pad = dest_size.width - src_size.width, height_pad = dest_size.height - src_size.height;
		size_t left = width_pad >> 1, right = width_pad - left;
		size_t top = height_pad >> 1, bottom = height_pad - top;
		cv::copyMakeBorder(dest, dest, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT, CV_RGB(0, 0, 0));

		return { ratio, left, right, top, bottom };
	}

	~transformation() noexcept = default;

	transformation(const transformation &) noexcept = default;
	transformation(transformation &&) = delete;

	transformation & operator=(const transformation &) = delete;
	transformation & operator=(transformation &&) = delete;

	template<typename Tensor>
		requires std::is_object_v<Tensor>
	void rescale(Tensor & boxes, const cv::Size & size) const;
private:
	inline transformation(
		double ratio = 0.0,
		int64_t left = 0,
		int64_t right = 0,
		int64_t top = 0,
		int64_t bottom = 0
	) noexcept : ratio(ratio), left(left), right(right), top(top), bottom(bottom) {}
};

}

#endif
