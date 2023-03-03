#ifndef __UTILITIES_CAMERA_FAKE_HPP__
#define __UTILITIES_CAMERA_FAKE_HPP__

#include <vector>
#include <system_error>

#include <opencv2/core.hpp>

#include "utilities/camera/base.hpp"

namespace utilities::camera
{

class fake final : public base::reader
{
public:
	[[nodiscard]]
	virtual bool next_image(std::error_code& ec, cv::Mat& image) override;
};

}

#endif
