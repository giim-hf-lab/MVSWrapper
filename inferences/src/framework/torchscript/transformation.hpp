#ifndef _INFERENCES_ENGINES_INTERNALS__INFERENCE_TORCHSCRIPT_TRANSFORMATION_HPP_
#define _INFERENCES_ENGINES_INTERNALS__INFERENCE_TORCHSCRIPT_TRANSFORMATION_HPP_

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include "../transformation.hpp"

namespace inferences
{

namespace framework::torchscript
{

namespace
{

inline static void xywh2xyxy(const torch::Tensor& boxes, const cv::Size& size)
{
	auto sx = boxes.select(2, 2) / 2;
	auto sy = boxes.select(2, 3) / 2;
	auto cx = boxes.select(2, 0);
	auto cy = boxes.select(2, 1);
	boxes.select(2, 2) = (cx + sx).clamp(0, size.width);
	boxes.select(2, 3) = (cy + sy).clamp(0, size.height);
	boxes.select(2, 0) = (cx - sx).clamp(0, size.width);
	boxes.select(2, 1) = (cy - sy).clamp(0, size.height);
}

}

}

namespace
{

template<>
void transformation::rescale<torch::Tensor>(torch::Tensor& boxes, const cv::Size& size) const
{
	if (ratio == 0.0)
		return;

	boxes.select(2, 0) = ((boxes.select(2, 0) - left) / ratio).clamp(0, size.width);
	boxes.select(2, 1) = ((boxes.select(2, 1) - top) / ratio).clamp(0, size.height);
	boxes.select(2, 2) = ((boxes.select(2, 2) - right) / ratio).clamp(0, size.width);
	boxes.select(2, 3) = ((boxes.select(2, 3) - bottom) / ratio).clamp(0, size.width);
}

}

}

#endif
