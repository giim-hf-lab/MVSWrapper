#ifndef __UTILITIES_STD_WIN_HPP__
#define __UTILITIES_STD_WIN_HPP__

#include <string>
#include <string_view>
#include <system_error>

#include "utilities/preprocessor.hpp"

namespace std
{

void from_string(
	error_code& ec,
	const string_view& str,
	wstring& out,
	bool native = true,
	bool low_memory = false
) noexcept;

inline void from_string(const string_view& str, wstring& out, bool native = true, bool low_memory = false)
{
	error_code ec;
	from_string(ec, str, out, native, low_memory);
	if (ec)
		throw system_error(ec);
}

UTILITIES_NODISCARD
inline wstring from_string(const string_view& str, bool native = true, bool low_memory = false)
{
	wstring out;
	from_string(str, out, native, low_memory);
	return out;
}

void to_string(
	error_code& ec,
	const wstring_view& str,
	string& out,
	bool native = true,
	bool low_memory = false
) noexcept;

inline void to_string(const wstring_view& str, string& out, bool native = true, bool low_memory = false)
{
	error_code ec;
	to_string(ec, str, out, native, low_memory);
	if (ec)
		throw system_error(ec);
}

UTILITIES_NODISCARD
inline string to_string(const wstring_view& str, bool native = true, bool low_memory = false)
{
	string out;
	to_string(str, out, native, low_memory);
	return out;
}

}

#endif
