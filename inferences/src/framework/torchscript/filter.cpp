#include <torch/torch.h>

#include "framework/torchscript/filter.hpp"

namespace inferences::framework::torchscript
{

filter::filter(torch::Tensor inclusion, torch::Tensor exclusion) :
	inclusion(std::move(inclusion)),
	exclusion(std::move(exclusion)) {}

filter::~filter() noexcept = default;

filter::filter(filter&&) noexcept = default;

}
