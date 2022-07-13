#ifndef __AUGMENTATION_HPP__
#define __AUGMENTATION_HPP__

#include <cstddef>

#include <algorithm>
#include <tuple>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace augmentation
{

[[nodiscard]]
auto letterbox(cv::Mat image, const cv::Size & new_size, bool scale_up, const cv::Scalar & padded_colour)
{
	auto image_size = image.size();
	auto ratio = std::min(double(new_size.width) / image_size.width, double(new_size.height) / image_size.height);

	cv::Mat tmp;
	if (ratio < 1.0 || (ratio > 1.0 && scale_up))
	{
		cv::resize(
			image,
			tmp,
			{ image_size.width * ratio, image_size.height * ratio },
			ratio,
			ratio,
			cv::InterpolationFlags::INTER_LINEAR
		);
		image = std::move(tmp);
	}
	else
		ratio = 1.0;

	image_size = image.size();

	size_t width_pad = new_size.width - image_size.width, height_pad = new_size.height - image_size.height;
	size_t left = width_pad >> 1, right = width_pad - left;
	size_t top = height_pad >> 1, bottom = height_pad - top;
	cv::copyMakeBorder(image, tmp, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT, padded_colour);

	return std::tuple { std::move(tmp), ratio, left, right, top, bottom };
}

}

#endif
