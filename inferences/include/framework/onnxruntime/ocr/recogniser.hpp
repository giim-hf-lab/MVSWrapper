#ifndef INFERENCES_FRAMEWORK_ONNXRUNTIME_OCR_RECOGNISER_HPP
#define INFERENCES_FRAMEWORK_ONNXRUNTIME_OCR_RECOGNISER_HPP

#include <cstddef>

#include <string>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "../model.hpp"

namespace inferences::framework::onnxruntime::ocr
{

struct recogniser final
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
	std::vector<std::vector<char>> _dictionary;
public:
	recogniser(
		const std::string& model_path,
		const std::string& dictionary_path,
		const Ort::SessionOptions& options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	);

	recogniser(
		const std::string& model_path,
		const std::string& dictionary_path,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	);

	~recogniser() noexcept;

	recogniser(const recogniser&) = delete;

	recogniser(recogniser&&) noexcept;

	recogniser& operator=(const recogniser&) = delete;

	recogniser& operator=(recogniser&&) = delete;

	[[nodiscard]]
	std::vector<std::tuple<size_t, std::string, double>> operator()(
		const std::vector<cv::Mat>& fragments,
		const parameters& parameters
	) &;

	void warmup(const parameters& parameters) &;
};

}

#endif
