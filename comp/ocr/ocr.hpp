#ifdef __INTELLISENSE__

#pragma once

#include "./ocr.inc.h"

#define __OCR_EXPORT

#else

#define __OCR_EXPORT export

#endif

namespace std
{

__OCR_EXPORT
template<typename T>
concept arithmetic = is_arithmetic_v<T>;

}

namespace
{

class model final
{
	[[nodiscard]]
	static inline Ort::Session _create_session(
		const std::basic_string<ORTCHAR_T>& model_path,
		Ort::Env& env,
		bool use_cuda,
		bool optimise
	)
	{
		Ort::SessionOptions session_options;
		session_options
			.EnableCpuMemArena()
			.EnableMemPattern()
			.DisableProfiling();

		if (use_cuda)
		{
			OrtCUDAProviderOptions cuda_options;
			cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchDefault;
			cuda_options.gpu_mem_limit = size_t(1) << 32;
			cuda_options.arena_extend_strategy = 1;
			session_options.AppendExecutionProvider_CUDA(cuda_options);
		}

		if (optimise)
		{
			auto optimised_model_path = model_path + ORT_TSTR(".opt");
			session_options
				.SetOptimizedModelFilePath(optimised_model_path.c_str())
				.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
		}

		return { env, model_path.c_str(), session_options };
	}

	Ort::Env _env;
	Ort::Session _session;
	Ort::Allocator _allocator;
	size_t _inputs, _outputs;
	std::vector<Ort::AllocatedStringPtr> _names;
	std::vector<const char *> _input_names, _output_names;
public:
	model(const std::basic_string<ORTCHAR_T>& model_path, bool use_cuda, bool optimise) :
		_env(),
		_session(_create_session(model_path, _env, use_cuda, optimise)),
		_allocator(
			_session,
			Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)
		),
		_inputs(_session.GetInputCount()),
		_outputs(_session.GetOutputCount()),
		_names(),
		_input_names(),
		_output_names()
	{
		_names.reserve(_inputs + _outputs);
		_input_names.reserve(_inputs);
		for (size_t i = 0; i < _inputs; ++i)
			_input_names.emplace_back(_names.emplace_back(_session.GetInputNameAllocated(i, _allocator)).get());
		_output_names.reserve(_outputs);
		for (size_t i = 0; i < _outputs; ++i)
			_output_names.emplace_back(_names.emplace_back(_session.GetOutputNameAllocated(i, _allocator)).get());
	}

	void run(
		const Ort::Value *inputs,
		size_t input_num,
		Ort::Value *outputs,
		size_t output_num,
		const Ort::RunOptions& run_options = {}
	)
	{
		_session.Run(run_options, _input_names.data(), inputs, input_num, _output_names.data(), outputs, output_num);
	}

	inline void run(
		const Ort::Value& input,
		Ort::Value& output,
		const Ort::RunOptions& run_options = {}
	)
	{
		run(&input, 1, &output, 1, run_options);
	}

	inline void run(
		const std::ranges::contiguous_range auto& inputs,
		std::ranges::contiguous_range auto& outputs,
		const Ort::RunOptions& run_options = {}
	)
	{
		run(
			std::ranges::data(inputs),
			std::ranges::size(inputs),
			std::ranges::data(outputs),
			std::ranges::size(outputs),
			run_options
		);
	}

	template<typename T>
	[[nodiscard]]
	Ort::Value tensor(const int64_t *shape, size_t shape_num)
	{
		return Ort::Value::CreateTensor<T>(_allocator, shape, shape_num);
	}

	template<typename T>
	[[nodiscard]]
	inline Ort::Value tensor(const std::ranges::contiguous_range auto& shape)
	{
		return tensor<T>(std::ranges::cdata(shape), std::ranges::size(shape));
	}

	template<typename T>
	[[nodiscard]]
	inline Ort::Value tensor(std::integral auto... shapes)
	{
		return tensor<T>(std::array { int64_t(shapes)... });
	}
};

static const auto _MASKED = cv::Scalar::all(0), _UNMASKED = cv::Scalar::all(1);

// static const auto _BLACK = cv::Scalar::all(0), _WHITE = cv::Scalar::all(255);

static const auto _kernel_2x2 = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, { 2, 2 });

}

namespace ocr
{

namespace scalers
{

__OCR_EXPORT
struct base
{
	virtual ~base() noexcept = default;

	virtual cv::Mat operator()(const cv::Mat& image) = 0;
};

__OCR_EXPORT
struct trivial final : public base
{
	[[nodiscard]]
	virtual cv::Mat operator()(const cv::Mat& image) noexcept override
	{
		return image;
	}
};

__OCR_EXPORT
class resize final : public base
{
	cv::Size _size;
public:
	resize(cv::Size size) noexcept : _size(std::move(size)) {}

	virtual cv::Mat operator()(const cv::Mat& image) override
	{
		if (image.size() == _size)
			return image;
		cv::Mat resized;
		cv::resize(image, resized, _size);
		return resized;
	}
};

__OCR_EXPORT
class zoom final : public base
{
	size_t _side_len;
	bool _min_side;
public:
	zoom(size_t side_len, bool min_side) noexcept : _side_len(side_len), _min_side(min_side) {}

	[[nodiscard]]
	virtual cv::Mat operator()(const cv::Mat& image) override
	{
		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/data/imaug/operators.py#L254-L301

		auto height = image.rows, width = image.cols;
		bool scaled;
		if (
			_min_side and height <= width and height < _side_len or
			not _min_side and height >= width and height > _side_len
		)
		{
			width *= double(_side_len) / height;
			height = _side_len;
			scaled = true;
		}
		else if (
			_min_side and height > width and width < _side_len or
			not _min_side and height < width and width > _side_len
		)
		{
			height *= double(_side_len) / width;
			width = _side_len;
			scaled = true;
		}
		else
			scaled = false;
		if (auto r = height % 32)
		{
			height += 32 - r;
			scaled = true;
		}
		if (auto r = width % 32)
		{
			width += 32 - r;
			scaled = true;
		}

		if (scaled)
		{
			cv::Mat resized;
			cv::resize(image, resized, { width, height });
			return resized;
		}
		return image;
	}
};

}

__OCR_EXPORT
template<typename T>
concept scaler = std::derived_from<T, scalers::base>;

namespace detectors
{

__OCR_EXPORT
struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual std::vector<cv::RotatedRect> operator()(const cv::Mat& image) = 0;
};

__OCR_EXPORT
struct trivial final : public base
{
	[[nodiscard]]
	virtual std::vector<cv::RotatedRect> operator()(const cv::Mat&) noexcept override
	{
		return {};
	}
};

__OCR_EXPORT
struct db final : public base
{
	enum class scoring_methods
	{
		BOX,
		APPROXIMATE,
		ORIGINAL
	};
private:
	cv::Scalar _mean, _stddev;
	double _threshold;
	bool _use_dilation;
	scoring_methods _scoring_method;
	size_t _max_candidates;
	double _box_threshold, _unclip_ratio, _min_fragment_ratio;
	model _model;
public:
	db(
		cv::Scalar mean,
		cv::Scalar stddev,
		double threshold,
		bool use_dilation,
		scoring_methods scoring_method,
		size_t max_candidates,
		double box_threshold,
		double unclip_ratio,
		double min_fragment_ratio,
		const std::basic_string<ORTCHAR_T>& model_path,
		bool use_cuda = true,
		bool optimise = true
	) :
		_mean(std::move(mean)),
		_stddev(std::move(stddev)),
		_threshold(threshold),
		_use_dilation(use_dilation),
		_scoring_method(scoring_method),
		_max_candidates(max_candidates),
		_box_threshold(box_threshold),
		_unclip_ratio(unclip_ratio),
		_min_fragment_ratio(min_fragment_ratio),
		_model(model_path, use_cuda, optimise)
	{}

	[[nodiscard]]
	virtual std::vector<cv::RotatedRect> operator()(const cv::Mat& image) override
	{
		const cv::Mat normalised(image.rows, image.cols, CV_32FC3);
		image.convertTo(normalised, CV_32FC3, 1 / 255.0);
		cv::subtract(normalised, _mean, normalised, cv::noArray());
		cv::divide(normalised, _stddev, normalised);

		auto input_tensor = _model.tensor<float>(1, 3, image.rows, image.cols);

		const auto stride = image.rows * image.cols;
		const auto input_data = input_tensor.GetTensorMutableData<float>();
		const auto split = std::array {
			cv::Mat { image.rows, image.cols, CV_32FC1, input_data },
			cv::Mat { image.rows, image.cols, CV_32FC1, input_data + stride },
			cv::Mat { image.rows, image.cols, CV_32FC1, input_data + 2 * stride }
		};
		cv::split(normalised, split);

		auto output_tensor = _model.tensor<float>(1, 1, image.rows, image.cols);

		_model.run(input_tensor, output_tensor);

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L230-L235

		const cv::Mat output(image.rows, image.cols, CV_32FC1, output_tensor.GetTensorMutableData<float>());
		const cv::Mat bitmap(image.rows, image.cols, CV_8UC1);
		cv::compare(output, _threshold, bitmap, cv::CmpTypes::CMP_GT);
		if (_use_dilation)
			cv::dilate(bitmap, bitmap, _kernel_2x2);

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L57-L149

		std::vector<cv::Mat> contours;
		cv::findContours(
			bitmap,
			contours,
			cv::RetrievalModes::RETR_LIST,
			cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE
		);

		size_t kept = 0;
		for (size_t i = 0; i < contours.size() and kept < _max_candidates; ++i)
		{
			auto& contour = contours[i];

			if (_scoring_method == scoring_methods::APPROXIMATE)
			{
				cv::approxPolyDP(contour, contour, 0.002 * cv::arcLength(contour, true), true);
				if (contour.rows < 4)
					continue;
			}

			// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L182-L218

			auto bounding = cv::boundingRect(contour);
			const cv::Mat mask(bounding.height, bounding.width, CV_8UC1, _MASKED);
			if (_scoring_method == scoring_methods::BOX)
			{
				cv::Mat vertices(4, 1, CV_32FC2);
				cv::minAreaRect(contour).points(vertices.ptr<cv::Point2f>());
				cv::fillPoly(mask, vertices, _UNMASKED, cv::LineTypes::LINE_AA, 0, -bounding.tl());
			}
			else
				cv::fillPoly(mask, contour, _UNMASKED, cv::LineTypes::LINE_AA, 0, -bounding.tl());

			if (cv::mean(output(bounding), mask)[0] < _box_threshold)
				continue;

			// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L151-L157

			Clipper2Lib::Path64 contour_path;
			contour_path.reserve(contour.rows);
			for (int i = 0; i < contour.rows; ++i)
			{
				const auto& op = contour.at<cv::Point>(i);
				contour_path.emplace_back(op.x, op.y);
			}
			Clipper2Lib::ClipperOffset clipper_offset;
			clipper_offset.AddPaths(
				{ std::move(contour_path) },
				Clipper2Lib::JoinType::Round,
				Clipper2Lib::EndType::Polygon
			);
			auto result = clipper_offset.Execute(
				cv::contourArea(contour) * _unclip_ratio / cv::arcLength(contour, true)
			);
			if (result.size() != 1)
				continue;

			auto& dest = contours[kept++];
			const auto& unclipped = result.front();
			dest.create(unclipped.size(), 1, CV_32SC2);
			for (size_t i = 0; i < unclipped.size(); ++i)
			{
				const auto& up = unclipped[i];
				auto& dp = dest.at<cv::Point>(i);
				dp.x = up.x;
				dp.y = up.y;
			}
		}

		if (not kept)
			return {};
		contours.resize(kept);

		std::vector<cv::RotatedRect> boxes;
		boxes.reserve(kept);
		for (const auto& contour : contours)
		{
			auto& box = boxes.emplace_back(cv::minAreaRect(contour));
			auto& angle = box.angle;
			if (angle < 0)
				angle += (size_t(-angle / 180) + 1) * 180;
			else if (angle > 0)
				angle -= size_t(angle / 180) * 180;
			auto& size = box.size;
			if (size.width > size.height * _min_fragment_ratio)
			{
				if (angle > 90)
					angle -= 180;
			}
			else
			{
				if (angle > 0)
					angle -= 90;
				else
					angle = 90;
				std::swap(size.width, size.height);
			}
		}
		return boxes;
	}
};

}

__OCR_EXPORT
template<typename T>
concept detector = std::derived_from<T, detectors::base>;

namespace classifiers
{

__OCR_EXPORT
struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual std::vector<size_t> operator()(const std::vector<cv::Mat>& fragments) = 0;
};

__OCR_EXPORT
struct trivial final : public base
{
	[[nodiscard]]
	virtual std::vector<size_t> operator()(const std::vector<cv::Mat>&) noexcept override
	{
		return {};
	}
};

__OCR_EXPORT
class concrete final : public base
{
	cv::Scalar _mean, _stddev;
	size_t _batch_size;
	cv::Size _shape;
	double _threshold;
	model _model;
public:
	concrete(
		cv::Scalar mean,
		cv::Scalar stddev,
		size_t batch_size,
		cv::Size shape,
		double threshold,
		const std::basic_string<ORTCHAR_T>& model_path,
		bool use_cuda = true,
		bool optimise = true
	) :
		_mean(std::move(mean)),
		_stddev(std::move(stddev)),
		_batch_size(batch_size),
		_shape(std::move(shape)),
		_threshold(threshold),
		_model(model_path, use_cuda, optimise)
	{}

	[[nodiscard]]
	virtual std::vector<size_t> operator()(const std::vector<cv::Mat>& fragments) override
	{
		auto input_tensor = _model.tensor<float>(_batch_size, 3, _shape.height, _shape.width);
		auto output_tensor = _model.tensor<float>(_batch_size, 2);

		const auto stride = _shape.height * _shape.width;
		const auto input_data = input_tensor.GetTensorMutableData<float>();

		std::vector<size_t> indices;
		indices.reserve(fragments.size());
		for (size_t i = 0; i < fragments.size(); i += _batch_size)
		{
			const size_t current_batch = std::min(_batch_size, fragments.size() - i);

			for (size_t b = 0, j = i; b < current_batch; ++b, ++j)
			{
				const cv::Mat normalised(_shape.height, _shape.width, CV_32FC3);

				if (const auto& fragment = fragments[j]; fragment.size() == _shape)
					fragment.convertTo(normalised, CV_32FC3, 1 / 255.0);
				else
				{
					const cv::Mat resized(_shape.height, _shape.width, CV_8UC3);
					cv::resize(fragment, resized, _shape);
					resized.convertTo(normalised, CV_32FC3, 1 / 255.0);
				}

				cv::subtract(normalised, _mean, normalised, cv::noArray());
				cv::divide(normalised, _stddev, normalised);

				const auto current_input_data = input_data + b * 3 * stride;
				const auto split = std::array {
					cv::Mat { _shape.height, _shape.width, CV_32FC1, current_input_data },
					cv::Mat { _shape.height, _shape.width, CV_32FC1, current_input_data + stride },
					cv::Mat { _shape.height, _shape.width, CV_32FC1, current_input_data + 2 * stride }
				};
				cv::split(normalised, split);
			}

			_model.run(input_tensor, output_tensor);

			const auto output_data = reinterpret_cast<const float (*)[2]>(output_tensor.GetTensorData<float>());
			for (size_t b = 0, j = i; b < current_batch; ++b, ++j)
			{
				const auto& [s0, s1] = output_data[b];
				if (s0 < s1 and s1 > _threshold)
					indices.push_back(j);
			}
		}
		return indices;
	}
};

}

__OCR_EXPORT
template<typename T>
concept classifier = std::derived_from<T, classifiers::base>;

namespace recognisers
{

struct recognition final
{
	size_t index;
	std::string text;
	double score;

	recognition(size_t index, std::string text, double score) :
		index(index),
		text(std::move(text)),
		score(score)
	{}
};

__OCR_EXPORT
struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual std::vector<recognition> operator()(const std::vector<cv::Mat>& fragments) = 0;
};

__OCR_EXPORT
struct trivial final : public base
{
	[[nodiscard]]
	virtual std::vector<recognition> operator()(const std::vector<cv::Mat>&) noexcept override
	{
		return {};
	}
};

__OCR_EXPORT
class ctc final : public base
{
	cv::Scalar _mean, _stddev;
	size_t _batch_size;
	cv::Size _shape;
	double _threshold;
	model _model;
	std::vector<std::vector<char>> _dictionary;
public:
	ctc(
		cv::Scalar mean,
		cv::Scalar stddev,
		size_t batch_size,
		cv::Size shape,
		double threshold,
		const std::string& dictionary_path,
		const std::basic_string<ORTCHAR_T>& model_path,
		bool use_cuda = true,
		bool optimise = true
	) :
		_mean(std::move(mean)),
		_stddev(std::move(stddev)),
		_batch_size(batch_size),
		_shape(std::move(shape)),
		_threshold(threshold),
		_model(model_path, use_cuda, optimise),
		_dictionary()
	{
		mio::mmap_source dict(dictionary_path);
		_dictionary.emplace_back();
		std::vector<char> buffer;
		buffer.reserve(4);
		for (size_t i = 0; i < dict.size(); ++i)
		{
			char c = dict[i];
			switch (c)
			{
				case '\r':
				case '\n':
					if (buffer.size())
					{
						_dictionary.push_back(buffer);
						buffer.clear();
					}
					break;
				default:
					buffer.push_back(c);
			}
		}
		if (buffer.size())
			_dictionary.emplace_back(std::move(buffer)).shrink_to_fit();
		_dictionary.emplace_back(1, ' ');
		_dictionary.shrink_to_fit();
	}

	[[nodiscard]]
	virtual std::vector<recognition> operator()(const std::vector<cv::Mat>& fragments) override
	{
		auto input_tensor = _model.tensor<float>(_batch_size, 3, _shape.height, _shape.width);
		auto output_tensor = _model.tensor<float>(_batch_size, 40, _dictionary.size());

		const auto stride = _shape.height * _shape.width;
		const auto input_data = input_tensor.GetTensorMutableData<float>();

		std::vector<recognition> results;
		results.reserve(fragments.size());
		for (size_t i = 0; i < fragments.size(); i += _batch_size)
		{
			const size_t current_batch = std::min(_batch_size, fragments.size() - i);

			for (size_t b = 0, j = i; b < current_batch; ++b, ++j)
			{
				const cv::Mat normalised(_shape.height, _shape.width, CV_32FC3);

				if (const auto& fragment = fragments[j]; fragment.size() == _shape)
					fragment.convertTo(normalised, CV_32FC3, 1 / 255.0);
				else
				{
					const cv::Mat resized(_shape.height, _shape.width, CV_8UC3);
					cv::resize(fragment, resized, _shape);
					resized.convertTo(normalised, CV_32FC3, 1 / 255.0);
				}

				cv::subtract(normalised, _mean, normalised, cv::noArray());
				cv::divide(normalised, _stddev, normalised);

				const auto current_input_data = input_data + b * 3 * stride;
				const auto split = std::array {
					cv::Mat { _shape.height, _shape.width, CV_32FC1, current_input_data },
					cv::Mat { _shape.height, _shape.width, CV_32FC1, current_input_data + stride },
					cv::Mat { _shape.height, _shape.width, CV_32FC1, current_input_data + 2 * stride }
				};
				cv::split(normalised, split);
			}

			_model.run(input_tensor, output_tensor);

			const auto output_data = output_tensor.GetTensorData<float>();
			for (size_t b = 0, j = i; b < current_batch; ++b, ++j)
			{
				size_t count = 0;
				double total_score = 0;
				size_t last_index = _dictionary.size();
				const auto batch_data = output_data + b * 40 * _dictionary.size();
				std::string buffer;
				for (size_t r = 0; r < 40; ++r)
				{
					const auto row_data = batch_data + r * _dictionary.size();
					const auto max_ptr = std::max_element(row_data, row_data + _dictionary.size());
					size_t index = max_ptr - row_data;
					if (index and index != last_index)
					{
						++count;
						total_score += *max_ptr;
						const auto& word = _dictionary[index];
						buffer.append(word.data(), word.size());
					}
					last_index = index;
				}
				total_score /= count;
				if (total_score >= _threshold)
					results.emplace_back(j, std::move(buffer), total_score);
			}
		}
		return results;
	}
};

}

__OCR_EXPORT
template<typename T>
concept recogniser = std::derived_from<T, recognisers::base>;

__OCR_EXPORT
struct result final
{
	cv::Mat vertices;
	std::string text;
	double score;

	result(cv::Mat vertices, std::string text, double score) :
		vertices(std::move(vertices)),
		text(std::move(text)),
		score(score)
	{}
};

__OCR_EXPORT
template<
	scaler Scaler,
	detector Detector,
	classifier Classifier,
	recogniser Recogniser
>
class system final
{
	Scaler _scaler;
	Detector _detector;
	Classifier _classifier;
	Recogniser _recogniser;
public:
	system(
		Scaler scaler,
		Detector detector,
		Classifier classifier,
		Recogniser recogniser
	) :
		_scaler(std::move(scaler)),
		_detector(std::move(detector)),
		_classifier(std::move(classifier)),
		_recogniser(std::move(recogniser))
	{}

	[[nodiscard]]
	std::vector<cv::RotatedRect> detect(const cv::Mat& image)
	{
		cv::Mat scaled = _scaler(image);
		auto boxes = _detector(scaled);
		if (boxes.empty())
			return {};
		auto width_ratio = double(image.cols) / scaled.cols, height_ratio = double(image.rows) / scaled.rows;
		for (auto& box : boxes)
		{
			box.center.x *= width_ratio;
			box.size.width *= width_ratio;
			box.center.y *= height_ratio;
			box.size.height *= height_ratio;
		}
		return boxes;
	}

	[[nodsicard]]
	std::vector<size_t> classify(const std::vector<cv::Mat>& images)
	{
		return _classifier(images);
	}

	[[nodiscard]]
	std::vector<recognisers::recognition> recognise(const std::vector<cv::Mat>& images)
	{
		return _recogniser(images);
	}

	[[nodiscard]]
	std::vector<result> operator()(const cv::Mat& image)
	{
		auto boxes = detect(image);

		std::vector<cv::Mat> fragments;
		fragments.reserve(boxes.size());
		for (const auto& box : boxes)
		{
			std::string coordinates;
			std::array<cv::Point2f, 4> contour;
			box.points(contour.data());
			const auto& size = box.size;
			auto vertices = std::array {
				cv::Point2f { 0, size.height },
				cv::Point2f { 0, 0 },
				cv::Point2f { size.width, 0 }
			};

			cv::warpAffine(
				image,
				fragments.emplace_back(),
				cv::getAffineTransform(contour.data(), vertices.data()),
				size,
				cv::InterpolationFlags::INTER_CUBIC,
				cv::BorderTypes::BORDER_REPLICATE
			);
		}

		for (auto index : classify(fragments))
		{
			const auto& fragment = fragments[index];
			cv::flip(fragment, fragment, -1);
		}

		auto recognitions = recognise(fragments);
		std::vector<result> results;
		results.reserve(recognitions.size());
		for (auto& [index, text, score] : recognitions)
		{
			const auto& box = boxes[index];
			cv::Mat vertices(4, 1, CV_32FC2);
			box.points(vertices.ptr<cv::Point2f>());
			const cv::Mat points(4, 1, CV_32SC2);
			vertices.convertTo(points, CV_32SC2);
			results.emplace_back(std::move(points), std::move(text), score);
		}
		return results;
	}
};

}
