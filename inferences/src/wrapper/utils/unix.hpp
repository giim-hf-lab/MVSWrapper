#ifndef _INFERENCES_WRAPPER_UTILS_UNIX_HPP_
#define _INFERENCES_WRAPPER_UTILS_UNIX_HPP_

#include <cerrno>

#include <system_error>
#include <type_traits>
#include <utility>

namespace inferences::wrapper::utils
{

namespace
{

template<typename F, typename... Args>
	requires std::is_invocable_r_v<int, F, Args...>
inline static int _wrap_unix(F&& call, Args&&... args)
{
	if (int ret = call(std::forward<Args>(args)...); ret < 0)
	[[unlikely]]
		throw std::system_error(errno, std::system_category(), "system operation failed");
	else
	[[likely]]
		return ret;
}

}

}

#endif
