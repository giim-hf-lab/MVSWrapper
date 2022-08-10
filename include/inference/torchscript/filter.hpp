#ifndef __INFERENCE_TORCHSCRIPT_FILTER_HPP__
#define __INFERENCE_TORCHSCRIPT_FILTER_HPP__

#include <utility>

#include <torch/script.h>

namespace inference::torchscript
{

struct filter final
{
	torch::Tensor inclusion, exclusion;

	filter(torch::Tensor inclusion, torch::Tensor exclusion) :
		inclusion(std::move(inclusion)),
		exclusion(std::move(exclusion)) {}

	~filter() noexcept = default;

	filter(const filter &) = delete;
	filter(filter &&) noexcept = default;

	filter & operator=(const filter &) = delete;
	filter & operator=(filter &&) noexcept = default;
};

}

#endif
