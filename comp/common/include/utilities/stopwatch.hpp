#ifndef __UTILITIES_STOPWATCH_HPP__
#define __UTILITIES_STOPWATCH_HPP__

#include <chrono>

namespace utilities
{

class stopwatch final
{
	std::chrono::steady_clock::time_point _last;
public:
	stopwatch() noexcept;

	template<typename Rep, typename Period>
	void elapsed(std::chrono::duration<Rep, Period>& duration)
	{
		duration = std::chrono::duration_cast<std::chrono::duration<Rep, Period>>(std::chrono::steady_clock::now() - _last);
	}

	template<typename Period, typename Rep>
	void elapsed(Rep& output)
	{
		std::chrono::duration<Rep, Period> duration;
		elapsed(duration);
		output = duration.count();
	}

	void reset() noexcept;
};

}

#endif
