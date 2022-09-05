#ifndef __INFERENCE_CONSTANTS_HPP__
#define __INFERENCE_CONSTANTS_HPP__

#include <opencv2/core.hpp>

namespace cv
{

static const auto IMAGE_NET_MEAN = CV_RGB(0.485, 0.456, 0.406);
static const auto IMAGE_NET_STDDEV = CV_RGB(0.229, 0.224, 0.225);

}

#endif
