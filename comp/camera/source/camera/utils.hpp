#ifndef __UTILITIES_CAMERA_UTILS_HPP__
#define __UTILITIES_CAMERA_UTILS_HPP__

#ifndef _UTILITIES_USE_FMT
#include <format>
#endif
#include <stdexcept>
#include <type_traits>

#ifdef _UTILITIES_USE_FMT
#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/format.h>
#endif
#include <opencv2/core.hpp>

#include "utilities/camera/base.hpp"

#ifdef _UTILITIES_USE_FMT
#define _UTILITIES_FORMAT_STRING(_F, ...) fmt::format(FMT_COMPILE(_F), __VA_ARGS__)
#else
#define _UTILITIES_FORMAT_STRING(_F, ...) std::format(_F, __VA_ARGS__)
#endif

#define _UTILITIES_FUNCTION_TEMPLATE(FNAME, ARGSNAME, R) template< \
	typename FNAME, \
	typename... ARGSNAME\
> \
	requires std::is_invocable_r_v<R, FNAME, ARGSNAME...>

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
