#ifndef __UTILITIES_INI_HPP__
#define __UTILITIES_INI_HPP__

#ifndef MINI_CASE_SENSITIVE
#define MINI_CASE_SENSITIVE
#endif

#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

#include <mini/ini.h>

#include "utilities/preprocessor.hpp"
#include "utilities/std.hpp"

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

	UTILITIES_NODISCARD
	bool load(std::filesystem::path path);

	UTILITIES_NODISCARD
	bool loaded() const noexcept;

	UTILITIES_NODISCARD
	bool modified() const noexcept;

	UTILITIES_NODISCARD
	bool read(std::string section, std::string key, std::string& value) const;

	UTILITIES_NODISCARD
	inline bool read(std::string section, std::string key, std::filesystem::path& path) const
	{
		std::string value;
		if (!read(std::move(section), std::move(key), value))
			return false;
		path = std::move(value);
		return true;
	}

	UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, std::, integral)
	UTILITIES_NODISCARD
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

		std::error_code ec;
		std::from_string(ec, str, value, base);
		return !ec;
	}

	UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, std::, floating_point)
	UTILITIES_NODISCARD
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

		std::error_code ec;
		std::from_string(ec, str, value, fmt);
		return !ec;
	}

	UTILITIES_NODISCARD
	bool save();

	UTILITIES_NODISCARD
	bool write(std::string section, std::string key, std::string value);

	UTILITIES_NODISCARD
	inline bool write(std::string section, std::string key, const std::filesystem::path& path)
	{
		return write(std::move(section), std::move(key), path.string());
	}

	UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, std::, integral)
	UTILITIES_NODISCARD
	inline bool write(
		std::string section,
		std::string key,
		T value,
		int base = 10
	)
	{
		std::string str;
		std::error_code ec;
		std::to_string(ec, value, str, base);
		return !ec && write(std::move(section), std::move(key), std::move(str));
	}

	UTILITIES_SIMPLE_CONCEPT_TEMPLATE(T, std::, floating_point)
	UTILITIES_NODISCARD
	inline bool write(
		std::string section,
		std::string key,
		T value,
		int precision = 6,
		std::chars_format fmt = std::chars_format::fixed
	)
	{
		std::string str;
		std::error_code ec;
		std::to_string(ec, value, str, precision, fmt);
		return !ec && write(std::move(section), std::move(key), std::move(str));
	}
};

}

#endif
