#ifndef __INFERENCE_ONNXRUNTIME_MODEL_HPP__
#define __INFERENCE_ONNXRUNTIME_MODEL_HPP__

#include <cstddef>

#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

namespace inference::onnxruntime
{

class model final
{
	[[nodiscard]]
	[[using gnu : always_inline]]
	inline static auto _default_options(std::string model_path)
	{
		model_path.append(".opt");

		Ort::SessionOptions session_options;
		OrtCUDAProviderOptions cuda_provider_options;
		session_options
			.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED)
			.SetOptimizedModelFilePath(model_path.c_str())
			.EnableCpuMemArena()
			.EnableMemPattern()
			.DisableProfiling()
			.AppendExecutionProvider_CUDA(cuda_provider_options);
		return session_options;
	}

	Ort::Env _env;
	Ort::Session _session;
	size_t _input_num, _output_num;
	std::vector<Ort::AllocatedStringPtr> _names_holders;
	std::vector<const char *> _input_names, _output_names;
public:
	model(const std::string & model_path, OrtAllocator * allocator, const Ort::SessionOptions & session_options) :
		_env(),
		_session(_env, model_path.c_str(), session_options),
		_input_num(_session.GetInputCount()),
		_output_num(_session.GetOutputCount()),
		_names_holders(),
		_input_names(),
		_output_names()
	{
		_names_holders.reserve(_input_num + _output_num);
		_input_names.reserve(_input_num);
		for (size_t i = 0; i < _input_num; ++i)
			_input_names.emplace_back(
				_names_holders.emplace_back(_session.GetInputNameAllocated(i, allocator)).get()
			);
		_output_names.reserve(_output_num);
		for (size_t i = 0; i < _output_num; ++i)
			_output_names.emplace_back(
				_names_holders.emplace_back(_session.GetOutputNameAllocated(i, allocator)).get()
			);
	}

	[[using gnu : always_inline]]
	inline model(const std::string & model_path, OrtAllocator * allocator) :
		model(model_path, allocator, _default_options(model_path)) {}

	void operator()(
		const Ort::Value * inputs,
		size_t input_num,
		Ort::Value * outputs,
		size_t output_num,
		const Ort::RunOptions & run_options = {}
	)
	{
		if (input_num != _input_num)
		[[unlikely]]
			throw std::runtime_error("model input number mismatch");
		if (output_num != _output_num)
		[[unlikely]]
			throw std::runtime_error("model output number mismatch");

		_session.Run(run_options, _input_names.data(), inputs, input_num, _output_names.data(), outputs, output_num);
	}

	[[using gnu : always_inline]]
	inline void operator()(const Ort::Value & input, Ort::Value & output, const Ort::RunOptions & run_options = {})
	{
		operator()(&input, 1, &output, 1, run_options);
	}

	template<std::ranges::contiguous_range InputRange, std::ranges::contiguous_range OutputRange>
	[[using gnu : always_inline]]
	inline void operator()(const InputRange & inputs, OutputRange & outputs, const Ort::RunOptions & run_options = {})
	{
		operator()(
			std::ranges::data(inputs),
			std::ranges::size(inputs),
			std::ranges::data(outputs),
			std::ranges::size(outputs),
			run_options
		);
	}
};

}

#endif
