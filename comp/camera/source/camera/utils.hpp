#ifndef __UTILITIES_CAMERA_UTILS_HPP__
#define __UTILITIES_CAMERA_UTILS_HPP__

#include <stdexcept>

#include <opencv2/core.hpp>

#include "utilities/camera/base.hpp"

namespace utilities::camera::_utils
{

namespace
{

static inline void rotate(const cv::Mat& image, cv::Mat& output, base::rotation_direction direction)
{
	switch (direction)
	{
		case base::rotation_direction::ORIGINAL:
			image.copyTo(output);
			break;
		case base::rotation_direction::CLOCKWISE_90:
			cv::transpose(image, output);
			cv::flip(output, output, 1);
			break;
		case base::rotation_direction::ANY_180:
			cv::flip(image, output, -1);
			break;
		case base::rotation_direction::COUNTER_CLOCKWISE_90:
			cv::transpose(image, output);
			cv::flip(output, output, 0);
			break;
		default:
			throw std::invalid_argument("invalid rotation direction");
	}
}

}

}

#endif
