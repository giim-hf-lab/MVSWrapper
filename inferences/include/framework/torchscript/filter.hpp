#ifndef INFERENCES_FRAMEWORK_TORCHSCRIPT_FILTER_HPP
#define INFERENCES_FRAMEWORK_TORCHSCRIPT_FILTER_HPP

#include <utility>

#include <torch/torch.h>

namespace inferences::framework::torchscript
{

struct filter final
{
	torch::Tensor inclusion, exclusion;

	filter(torch::Tensor inclusion, torch::Tensor exclusion);

	~filter() noexcept;

	filter(const filter&) = delete;

	filter(filter&&) noexcept;

	filter& operator=(const filter&) = delete;

	filter& operator=(filter&&) = delete;
};

}

#endif
