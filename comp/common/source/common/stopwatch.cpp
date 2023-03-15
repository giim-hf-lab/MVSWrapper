#include "utilities/stopwatch.hpp"

namespace utilities
{

stopwatch::stopwatch() noexcept : _last { std::chrono::steady_clock::now() } {}

void stopwatch::reset() noexcept
{
	_last = std::chrono::steady_clock::now();
}

}
