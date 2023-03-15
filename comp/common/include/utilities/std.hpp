#ifndef __UTILITIES_STD_HPP__
#define __UTILITIES_STD_HPP__

#include <charconv>
#include <concepts>
#include <string>
#include <string_view>
#include <system_error>

#ifdef _WIN32
#  include "utilities/std/win.hpp"
#endif

namespace std
{

inline void from_string(error_code& ec, const string_view& str, integral auto& out, int base = 10)
{
	auto [ptr, eec] = from_chars(str.data(), str.data() + str.size(), out, base);
	if (eec != errc())
		ec = make_error_code(eec);
	else if (ptr - str.data() != str.size())
		ec = make_error_code(errc::invalid_argument);
	else
		ec.clear();
}

inline void from_string(const string_view& str, integral auto& out, int base = 10)
{
	error_code ec;
	from_string(ec, str, out, base);
	if (ec)
		throw system_error(ec);
}

template<integral T>
[[nodiscard]]
inline T from_string(const string_view& str, int base = 10)
{
	T out;
	from_string(str, out, base);
	return out;
}

inline void from_string(error_code& ec, const string_view& str, floating_point auto& out, chars_format fmt = chars_format::general)
{
	auto [ptr, eec] = from_chars(str.data(), str.data() + str.size(), out, fmt);
	if (eec != errc())
		ec = make_error_code(eec);
	else if (ptr - str.data() != str.size())
		ec = make_error_code(errc::invalid_argument);
	else
		ec.clear();
}

inline void from_string(const string_view& str, floating_point auto& out, chars_format fmt = chars_format::general)
{
	error_code ec;
	from_string(ec, str, out, fmt);
	if (ec)
		throw system_error(ec);
}

template<floating_point T>
[[nodiscard]]
inline T from_string(const string_view& str, chars_format fmt = chars_format::general)
{
	T out;
	from_string(str, out, fmt);
	return out;
}

template<integral T>
inline void to_string(error_code& ec, T value, string& out, int base = 10)
{
	using limits = numeric_limits<T>;
	static constexpr size_t width = limits::is_signed + limits::digits;

	out.resize(width);
	auto [ptr, eec] = to_chars(out.data(), out.data() + out.size(), value, base);
	if (eec != errc())
		ec = make_error_code(eec);
	else
	{
		out.resize(ptr - out.data());
		ec.clear();
	}
}

inline void to_string(integral auto value, string& out, int base = 10)
{
	error_code ec;
	to_string(ec, value, out, base);
	if (ec)
		throw system_error(ec);
}

[[nodiscard]]
inline string to_string(integral auto value, int base)
{
	string out;
	to_string(value, out, base);
	return out;
}

template<floating_point T>
inline void to_string(error_code& ec, T value, string& out, int precision = 6, chars_format fmt = chars_format::fixed)
{
	using limits = numeric_limits<T>;
	static constexpr size_t fixed_width = limits::digits10 + limits::max_exponent10 + 4;

	out.resize(fixed_width + precision);
	auto [ptr, eec] = to_chars(out.data(), out.data() + out.size(), value, fmt, precision);
	if (eec != errc())
		ec = make_error_code(eec);
	else
	{
		out.resize(ptr - out.data());
		ec.clear();
	}
}

inline void to_string(floating_point auto value, string& out, int precision = 6, chars_format fmt = chars_format::fixed)
{
	error_code ec;
	to_string(ec, value, out, precision, fmt);
	if (ec)
		throw system_error(ec);
}

[[nodiscard]]
inline string to_string(floating_point auto value, int precision, chars_format fmt = chars_format::fixed)
{
	string out;
	to_string(value, out, precision, fmt);
	return out;
}

}

#endif
