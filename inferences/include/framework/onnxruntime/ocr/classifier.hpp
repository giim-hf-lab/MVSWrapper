#ifndef INFERENCES_FRAMEWORK_ONNXRUNTIME_OCR_CLASSIFIER_HPP
#define INFERENCES_FRAMEWORK_ONNXRUNTIME_OCR_CLASSIFIER_HPP

#include <cstddef>

#include <string>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "../model.hpp"

namespace inferences::framework::onnxruntime::ocr
{

struct classifier final
{
	struct parameters final
	{
		size_t batch_size;
		cv::Size shape;
		cv::Scalar mean, stddev;
		double threshold;

		parameters(
			size_t batch_size,
			const cv::Size& shape,
			const cv::Scalar& mean,
			const cv::Scalar& stddev,
			double threshold
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
	classifier(
		const std::string& model_path,
		const Ort::SessionOptions& options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	);

	classifier(
		const std::string& model_path,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	);

	~classifier() noexcept;

	classifier(const classifier&) = delete;

	classifier(classifier&&) noexcept;

	classifier& operator=(const classifier&) = delete;

	classifier& operator=(classifier&&) = delete;

	[[nodiscard]]
	std::vector<cv::Mat> operator()(
		const cv::Mat& image,
		const std::vector<cv::RotatedRect>& boxes,
		const parameters& parameters
	) &;

	void warmup(const parameters& parameters) &;
};

}

#endif
