#pragma once

#ifndef MINI_CASE_SENSITIVE
#define MINI_CASE_SENSITIVE
#endif

#include <filesystem>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include <mini/ini.h>

#include "utilities/preprocessor.hpp"

namespace utilities
{

class ini final
{
	std::filesystem::path _path;
	mINI::INIStructure _data;
	bool _modified;
public:
	ini() noexcept;

	inline ini(std::filesystem::path path) : ini()
	{
		if (!load(std::move(path)))
			throw std::system_error(std::make_error_code(std::errc::no_such_file_or_directory));
	}

	inline ~ini() noexcept
	{
		if (modified() && !save())
			throw std::system_error(std::make_error_code(std::errc::no_such_file_or_directory));
	}

	[[nodiscard]]
	bool load(std::filesystem::path path);

	[[nodiscard]]
	bool loaded() const noexcept;

	[[nodiscard]]
	bool modified() const noexcept;

	[[nodiscard]]
	bool read(std::string section, std::string key, std::string& value) const;

	[[nodiscard]]
	inline bool read(std::string section, std::string key, std::filesystem::path& path) const
	{
		std::string value;
		if (!read(std::move(section), std::move(key), value))
			return false;
		path = std::move(value);
		return true;
	}

	UTILITIES_TEMPLATE_SIMPLE_CONSTRAINT(T, std::is_integral_v)
	[[nodiscard]]
	inline bool read(
		std::string section,
		std::string key,
		T& value,
		int base = 10
	) const
	{
		std::string str;
		if (!read(std::move(section), std::move(key), str))
			return false;

		auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), value, base);
		return ec == std::errc() && ptr - str.data() == str.size();
	}

	UTILITIES_TEMPLATE_SIMPLE_CONSTRAINT(T, std::is_floating_point_v)
	[[nodiscard]]
	inline bool read(
		std::string section,
		std::string key,
		T& value,
		std::chars_format fmt = std::chars_format::fixed
	) const
	{
		std::string str;
		if (!read(std::move(section), std::move(key), str))
			return false;

		auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), value, fmt);
		return ec == std::errc() && ptr - str.data() == str.size();
	}

	[[nodiscard]]
	bool save();

	[[nodiscard]]
	bool write(std::string section, std::string key, std::string value);

	[[nodiscard]]
	inline bool write(std::string section, std::string key, const std::filesystem::path& path)
	{
		return write(std::move(section), std::move(key), path.string());
	}

	UTILITIES_TEMPLATE_SIMPLE_CONSTRAINT(T, std::is_integral_v)
	[[nodiscard]]
	inline bool write(
		std::string section,
		std::string key,
		T value,
		int base = 10
	)
	{
		using limits = std::numeric_limits<T>;
		static constexpr size_t width = limits::is_signed + limits::digits;

		std::string str;
		str.resize(width);
		auto [ptr, ec] = std::to_chars(str.data(), str.data() + str.size(), value, base);
		if (ec != std::errc())
			return false;
		str.resize(ptr - str.data());

		return write(std::move(section), std::move(key), std::move(str));
	}

	UTILITIES_TEMPLATE_SIMPLE_CONSTRAINT(T, std::is_floating_point_v)
	[[nodiscard]]
	inline bool write(
		std::string section,
		std::string key,
		T value,
		int precision = 6,
		std::chars_format fmt = std::chars_format::fixed
	)
	{
		using limits = std::numeric_limits<T>;
		static constexpr size_t fixed_width = limits::digits10 + limits::max_exponent10 + 5;

		std::string str;
		str.resize(fixed_width + precision);
		auto [ptr, ec] = std::to_chars(str.data(), str.data() + str.size(), value, fmt, precision);
		if (ec != std::errc())
			return false;
		str.resize(ptr - str.data());

		return write(std::move(section), std::move(key), std::move(str));
	}
};

}
