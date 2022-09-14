#ifndef INFERENCES_FRAMEWORK_ONNXRUNTIME_MODEL_HPP
#define INFERENCES_FRAMEWORK_ONNXRUNTIME_MODEL_HPP

#include <cstddef>
#include <cstdint>

#include <ranges>
#include <string>

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

namespace inferences::framework::onnxruntime
{

class model final
{
	static Ort::AllocatorWithDefaultOptions _allocator;

	Ort::Env _env;
	Ort::Session _session;
	std::vector<Ort::AllocatedStringPtr> _names;
	std::vector<const char *> _input_names, _output_names;
public:
	template<typename T>
	[[nodiscard]]
	static Ort::Value tensor(const int64_t *shape, size_t shape_len)
	{
		return Ort::Value::CreateTensor<T>(_allocator, shape, shape_len);
	}

	model(
		const std::string& model_path,
		const Ort::SessionOptions& common_options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	);

	~model() noexcept;

	model(const model&) = delete;

	model(model&&) noexcept;

	model& operator=(const model&) = delete;

	model& operator=(model&&) = delete;

	void operator()(
		const Ort::Value *inputs,
		size_t input_num,
		Ort::Value *outputs,
		size_t output_num,
		const Ort::RunOptions& run_options = {}
	) &;

	inline void operator()(const Ort::Value& input, Ort::Value& output, const Ort::RunOptions& run_options = {}) &
	{
		operator()(&input, 1, &output, 1, run_options);
	}

	template<std::ranges::contiguous_range InputRange, std::ranges::contiguous_range OutputRange>
	inline void operator()(const InputRange& inputs, OutputRange& outputs, const Ort::RunOptions& run_options = {}) &
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
