#ifndef __UTILITIES_WINUTILS_STRINGS_HPP__
#define __UTILITIES_WINUTILS_STRINGS_HPP__

#include <string>
#include <string_view>
#include <system_error>

namespace std
{

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

[[nodiscard]]
inline string to_string(const wstring_view& str, bool native = true, bool low_memory = false)
{
	string out;
	to_string(str, out, native, low_memory);
	return out;
}

void to_wstring(
	error_code& ec,
	const string_view& str,
	wstring& out,
	bool native = true,
	bool low_memory = false
) noexcept;

inline void to_wstring(const string_view& str, wstring& out, bool native = true, bool low_memory = false)
{
	error_code ec;
	to_wstring(ec, str, out, native, low_memory);
	if (ec)
		throw system_error(ec);
}

[[nodiscard]]
inline wstring to_wstring(const string_view& str, bool native = true, bool low_memory = false)
{
	wstring out;
	to_wstring(str, out, native, low_memory);
	return out;
}

}

#endif
