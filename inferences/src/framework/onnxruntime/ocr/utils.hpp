#ifndef _INFERENCES_ENGINES_INTERNALS__INFERENCE_ONNXRUNTIME_OCR_UTILS_HPP_
#define _INFERENCES_ENGINES_INTERNALS__INFERENCE_ONNXRUNTIME_OCR_UTILS_HPP_

#include <cstddef>

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "../../transformation.hpp"

namespace inferences::framework::onnxruntime::ocr
{

namespace
{

[[nodiscard]]
inline static auto _default_options() noexcept
{
	Ort::SessionOptions session_options;
	OrtCUDAProviderOptions cuda_provider_options;
	session_options
		.EnableCpuMemArena()
		.EnableMemPattern()
		.DisableProfiling()
		.AppendExecutionProvider_CUDA(cuda_provider_options);
	return session_options;
}

static const auto _GLOBAL_DEFAULT_OPTIONS = _default_options();

inline static transformation _scale_split_image(
	const cv::Mat& image,
	const cv::Size& shape,
	const cv::Scalar& mean,
	const cv::Scalar& stddev,
	float *output
)
{
	cv::Mat transformed;
	auto scaler = transformation::letterbox(image, transformed, shape, true, mean, stddev);

	size_t stride = shape.area();
	cv::Mat split[] {
		{ shape, CV_32FC1, output },
		{ shape, CV_32FC1, output + stride },
		{ shape, CV_32FC1, output + 2 * stride }
	};
	cv::split(transformed, split);

	return scaler;
}

}

}

#endif
