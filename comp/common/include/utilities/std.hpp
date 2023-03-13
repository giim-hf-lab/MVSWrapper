#ifndef __UTILITIES_STD_HPP__
#define __UTILITIES_STD_HPP__

#include <charconv>
#include <string>
#include <string_view>
#include <system_error>

#include "utilities/preprocessor.hpp"

#ifdef _WIN32
#  include "utilities/std/win.hpp"
#endif

namespace std
{

#if __cpp_lib_concepts >= 202002L
template<typename T>
concept arithmetic = is_arithmetic_v<T>;
#endif

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , integral)
inline void from_string(error_code& ec, const string_view& str, T& out, int base = 10)
{
	auto [ptr, ec] = from_chars(str.data(), str.data() + str.size(), out, base);
	if (ec != errc())
		ec = make_error_code(ec);
	else if (ptr - str.data() != str.size())
		ec = make_error_code(errc::invalid_argument);
	else
		ec.clear();
}

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , integral)
inline void from_string(const string_view& str, T& out, int base = 10)
{
	error_code ec;
	from_string(ec, str, out, base);
	if (ec)
		throw system_error(ec);
}

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , integral)
UTILITIES_NODISCARD
inline T from_string(const string_view& str, int base = 10)
{
	T out;
	from_string(str, out, base);
	return out;
}

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , floating_point)
inline void from_string(error_code& ec, const string_view& str, T& out, chars_format fmt = chars_format::general)
{
	auto [ptr, ec] = from_chars(str.data(), str.data() + str.size(), out, fmt);
	if (ec != errc())
		ec = make_error_code(ec);
	else if (ptr - str.data() != str.size())
		ec = make_error_code(errc::invalid_argument);
	else
		ec.clear();
}

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , floating_point)
inline void from_string(const string_view& str, T& out, chars_format fmt = chars_format::general)
{
	error_code ec;
	from_string(ec, str, out, fmt);
	if (ec)
		throw system_error(ec);
}

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , floating_point)
UTILITIES_NODISCARD
inline T from_string(const string_view& str, chars_format fmt = chars_format::general)
{
	T out;
	from_string(str, out, fmt);
	return out;
}

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , integral)
inline void to_string(error_code& ec, T value, string& out, int base = 10)
{
	using limits = numeric_limits<T>;
	static constexpr size_t width = limits::is_signed + limits::digits;

	out.resize(width);
	auto [ptr, ec] = to_chars(out.data(), out.data() + out.size(), value, base);
	if (ec != errc())
		ec = make_error_code(ec);
	else
		out.resize(ptr - out.data());
}

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , integral)
inline void to_string(T value, string& out, int base = 10)
{
	error_code ec;
	to_string(ec, value, out, base);
	if (ec)
		throw system_error(ec);
}

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , integral)
UTILITIES_NODISCARD
inline string to_string(T value, int base)
{
	string out;
	to_string(value, out, base);
	return out;
}

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , floating_point)
inline void to_string(error_code& ec, T value, string& out, int precision = 6, chars_format fmt = chars_format::fixed)
{
	using limits = numeric_limits<T>;
	static constexpr size_t fixed_width = limits::digits10 + limits::max_exponent10 + 4;

	out.resize(fixed_width + precision);
	auto [ptr, ec] = to_chars(out.data(), out.data() + out.size(), value, fmt, precision);
	if (ec != errc())
		ec = make_error_code(ec);
	else
		out.resize(ptr - out.data());
}

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , floating_point)
inline void to_string(T value, string& out, int precision = 6, chars_format fmt = chars_format::fixed)
{
	error_code ec;
	to_string(ec, value, out, precision, fmt);
	if (ec)
		throw system_error(ec);
}

UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, , floating_point)
UTILITIES_NODISCARD
inline string to_string(T value, int precision, chars_format fmt = chars_format::fixed)
{
	string out;
	to_string(value, out, precision, fmt);
	return out;
}

}

#endif
