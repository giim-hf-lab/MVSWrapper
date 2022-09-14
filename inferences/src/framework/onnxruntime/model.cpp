#include <cstddef>

#include <stdexcept>
#include <string>

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include "framework/onnxruntime/model.hpp"

namespace inferences::framework::onnxruntime
{

namespace
{

[[nodiscard]]
inline static Ort::Session _create_session(
	const std::string& model_path,
	Ort::Env& env,
	const Ort::SessionOptions& common_options,
	GraphOptimizationLevel graph_opt_level
)
{
	if (graph_opt_level > GraphOptimizationLevel::ORT_DISABLE_ALL)
	[[likely]]
	{
		auto optimised_model_path = model_path + ".opt";
		auto session_options = common_options.Clone();
		session_options
			.SetOptimizedModelFilePath(optimised_model_path.c_str())
			.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
		return { env, model_path.c_str(), session_options };
	}
	return { env, model_path.c_str(), common_options };
}

}

Ort::AllocatorWithDefaultOptions model::_allocator;

model::model(
	const std::string& model_path,
	const Ort::SessionOptions& common_options,
	GraphOptimizationLevel graph_opt_level
) :
	_env(),
	_session(_create_session(model_path, _env, common_options, graph_opt_level)),
	_names(),
	_input_names(),
	_output_names()
{
	auto input_num = _session.GetInputCount(), output_num = _session.GetOutputCount();

	_names.reserve(input_num + output_num);
	_input_names.reserve(input_num);
	for (size_t i = 0; i < input_num; ++i)
		_input_names.emplace_back(
			_names.emplace_back(_session.GetInputNameAllocated(i, _allocator)).get()
		);
	_output_names.reserve(output_num);
	for (size_t i = 0; i < output_num; ++i)
		_output_names.emplace_back(
			_names.emplace_back(_session.GetOutputNameAllocated(i, _allocator)).get()
		);
}

model::~model() noexcept = default;

model::model(model&&) noexcept = default;

void model::operator()(
	const Ort::Value *inputs,
	size_t input_num,
	Ort::Value *outputs,
	size_t output_num,
	const Ort::RunOptions& run_options
) &
{
	if (input_num != _input_names.size())
	[[unlikely]]
		throw std::runtime_error("model input number mismatch");
	if (output_num != _output_names.size())
	[[unlikely]]
		throw std::runtime_error("model output number mismatch");

	_session.Run(run_options, _input_names.data(), inputs, input_num, _output_names.data(), outputs, output_num);
}

}
