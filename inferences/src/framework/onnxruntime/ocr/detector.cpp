#include <cstddef>

#include <string>
#include <vector>

#include <clipper.core.h>
#include <clipper.offset.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "framework/onnxruntime/ocr/detector.hpp"

#include "./utils.hpp"

namespace
{

[[nodiscard]]
inline static auto _binarise_scores(const cv::Mat& scores, double threshold, bool dilation)
{
	cv::Mat bitmap;
	cv::threshold(scores, bitmap, threshold, 255, cv::ThresholdTypes::THRESH_BINARY);
	bitmap.convertTo(bitmap, CV_8UC1, 1.0, 0.0);
	if (dilation)
		cv::dilate(bitmap, bitmap, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, { 2, 2 }));
	return bitmap;
}

[[nodiscard]]
inline static auto _approximate_score(const cv::Mat& scores, const std::vector<cv::Point>& contour)
{
	auto enclosing = cv::minAreaRect(contour);
	auto bounding = enclosing.boundingRect();
	cv::Point2f corners[4];
	enclosing.points(corners);
	auto offset = bounding.tl();
	cv::Point points[4] {
		static_cast<cv::Point>(corners[0]) - offset,
		static_cast<cv::Point>(corners[1]) - offset,
		static_cast<cv::Point>(corners[2]) - offset,
		static_cast<cv::Point>(corners[3]) - offset
	};
	auto mask = cv::Mat(bounding.size(), CV_8UC1, cv::Scalar(0));
	// rectangle is always convex
	cv::fillConvexPoly(mask, points, 4, 255);
	return cv::mean(scores(bounding), mask)[0];
}

[[nodiscard]]
inline static auto _accurate_score(const cv::Mat& scores, const std::vector<cv::Point>& contour)
{
	auto bounding = cv::boundingRect(contour);
	auto mask = cv::Mat(bounding.size(), CV_8UC1, cv::Scalar(0));
	cv::fillPoly(mask, contour, 255, cv::LineTypes::LINE_8, 0, -bounding.tl());
	return cv::mean(scores(bounding), mask)[0];
}

[[nodiscard]]
inline static auto _enclose_enlarge(
	const cv::Mat& scores,
	std::vector<std::vector<cv::Point>>& contours,
	bool fast_scoring,
	double score_threshold,
	double unclip_ratio,
	double min_box_side_length
)
{
	std::vector<cv::RotatedRect> results;
	results.reserve(contours.size());
	for (size_t i = 0; i < contours.size(); ++i)
	{
		auto & contour = contours[i];
		if (contour.size() < 3)
		[[unlikely]]
			continue;

		if ((
			// (param.fast_scoring ? _approximate_score : _accurate_score)(scores, contour)
			fast_scoring ? _approximate_score(scores, contour) : _accurate_score(scores, contour)
		) < score_threshold)
			continue;

		auto offset = cv::contourArea(contour, false) * unclip_ratio / cv::arcLength(contour, true);

		Clipper2Lib::Path64 contour_path;
		contour_path.reserve(contour.size());
		for (const auto & point : contour)
			contour_path.emplace_back(point.x, point.y);

		Clipper2Lib::ClipperOffset clipper_offset;
		clipper_offset.AddPaths({ contour_path }, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
		contour_path = std::move(clipper_offset.Execute(offset)[0]);
		contour.clear();
		contour.reserve(contour_path.size());
		for (const auto & point : contour_path)
			contour.emplace_back(point.x, point.y);

		auto enclosing = cv::minAreaRect(contour);
		if (const auto & size = enclosing.size; std::min(size.height, size.width) <= min_box_side_length)
			continue;
		results.emplace_back(std::move(enclosing));
	}
	return results;
}

}

namespace inferences::framework::onnxruntime::ocr
{

detector::parameters::parameters(
	const cv::Size& shape,
	const cv::Scalar& mean,
	const cv::Scalar& stddev,
	double threshold,
	bool dilation,
	bool fast_scoring,
	double score_threshold,
	double unclip_ratio,
	double min_box_side_length
) noexcept :
	shape(shape),
	mean(mean),
	stddev(stddev),
	threshold(threshold),
	dilation(dilation),
	fast_scoring(fast_scoring),
	score_threshold(score_threshold),
	unclip_ratio(unclip_ratio),
	min_box_side_length(min_box_side_length) {}

detector::parameters::~parameters() noexcept = default;

detector::parameters::parameters(const parameters&) noexcept = default;

detector::parameters::parameters(parameters&&) noexcept = default;

detector::parameters& detector::parameters::operator=(const parameters&) noexcept = default;

detector::parameters& detector::parameters::operator=(parameters&&) noexcept = default;

detector::detector(
	const std::string& model_path,
	const Ort::SessionOptions& options,
	GraphOptimizationLevel graph_opt_level
) : _model(model_path, options, graph_opt_level) {}

detector::detector(
	const std::string& model_path,
	GraphOptimizationLevel graph_opt_level
) : detector(model_path, _GLOBAL_DEFAULT_OPTIONS, graph_opt_level) {}

detector::~detector() noexcept = default;

detector::detector(detector&&) noexcept = default;

[[nodiscard]]
std::vector<cv::RotatedRect> detector::operator()(const cv::Mat& image, const parameters& parameters) &
{
	int64_t input_shape[] { 1, 3, parameters.shape.height, parameters.shape.width };
	auto input_tensor = model::tensor<float>(input_shape, 4);
	auto scaler = _scale_split_image(
		image,
		parameters.shape,
		parameters.mean,
		parameters.stddev,
		input_tensor.GetTensorMutableData<float>()
	);
	int64_t output_shape[] { 1, 1, parameters.shape.height, parameters.shape.width };
	auto output_tensor = model::tensor<float>(output_shape, 4);

	_model(input_tensor, output_tensor);

	cv::Mat scores(parameters.shape, CV_32FC1, output_tensor.GetTensorMutableData<float>());
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(
		_binarise_scores(scores, parameters.threshold, parameters.dilation),
		contours,
		cv::RetrievalModes::RETR_LIST,
		cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE
	);

	auto results = _enclose_enlarge(
		scores,
		contours,
		parameters.fast_scoring,
		parameters.score_threshold,
		parameters.unclip_ratio,
		parameters.min_box_side_length * scaler.ratio
	);
	scaler.rescale(results, image.size());
	return results;
}

void detector::warmup(const parameters& parameters) &
{
	int64_t input_shape[] { 1, 3, parameters.shape.height, parameters.shape.width };
	auto input_tensor = model::tensor<float>(input_shape, 4);
	int64_t output_shape[] { 1, 1, parameters.shape.height, parameters.shape.width };
	auto output_tensor = model::tensor<float>(output_shape, 4);
	_model(input_tensor, output_tensor);
}

}
