#ifndef INFERENCES_FRAMEWORK_ONNXRUNTIME_OCR_DETECTOR_HPP
#define INFERENCES_FRAMEWORK_ONNXRUNTIME_OCR_DETECTOR_HPP

#include <cstddef>

#include <string>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "../model.hpp"

namespace inferences::framework::onnxruntime::ocr
{

struct detector final
{
	struct parameters final
	{
		cv::Size shape;
		cv::Scalar mean, stddev;
		double threshold;
		bool dilation, fast_scoring;
		double score_threshold, unclip_ratio, min_box_side_length;

		parameters(
			const cv::Size& shape,
			const cv::Scalar& mean,
			const cv::Scalar& stddev,
			double threshold,
			bool dilation,
			bool fast_scoring,
			double score_threshold,
			double unclip_ratio,
			double min_box_side_length
		) noexcept;

		~parameters() noexcept;

		parameters(const parameters&) noexcept;

		parameters(parameters&&) noexcept;

		parameters& operator=(const parameters&) noexcept;

		parameters& operator=(parameters&&) noexcept;
	};
private:
	model _model;
public:
	detector(
		const std::string& model_path,
		const Ort::SessionOptions& options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	);

	detector(
		const std::string& model_path,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	);

	~detector() noexcept;

	detector(const detector&) = delete;

	detector(detector&&) noexcept;

	detector& operator=(const detector&) = delete;

	detector& operator=(detector&&) = delete;

	[[nodiscard]]
	std::vector<cv::RotatedRect> operator()(const cv::Mat& image, const parameters& parameters) &;

	void warmup(const parameters& parameters) &;
};

}

#endif
