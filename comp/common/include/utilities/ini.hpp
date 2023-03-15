#ifndef __UTILITIES_INI_HPP__
#define __UTILITIES_INI_HPP__

#ifndef MINI_CASE_SENSITIVE
#define MINI_CASE_SENSITIVE
#endif

#include <concepts>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

#include <mini/ini.h>

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

	ini(std::filesystem::path path);

	inline ~ini() noexcept
	{
		if (modified())
			save();
	}

	[[nodiscard]]
	bool created() const noexcept;

	void load(std::error_code& ec, std::filesystem::path path);

	inline void load(std::filesystem::path path)
	{
		std::error_code ec;
		load(ec, std::move(path));
		if (ec)
			throw std::system_error(ec);
	}

	[[nodiscard]]
	bool modified() const noexcept;

	void read(std::error_code& ec, std::string section, std::string key, std::string& value) const;

	inline void read(std::string section, std::string key, std::string& value) const
	{
		std::error_code ec;
		read(ec, std::move(section), std::move(key), value);
		if (ec)
			throw std::system_error(ec);
	}

	inline void read(std::error_code& ec, std::string section, std::string key, std::filesystem::path& path) const
	{
		std::string value;
		read(ec, std::move(section), std::move(key), value);
		if (ec)
			return;
		path = std::move(value);
	}

	inline void read(std::string section, std::string key, std::filesystem::path& path) const
	{
		std::error_code ec;
		read(ec, std::move(section), std::move(key), path);
		if (ec)
			throw std::system_error(ec);
	}

	inline void read(
		std::error_code& ec,
		std::string section,
		std::string key,
		std::integral auto& value,
		int base = 10
	) const
	{
		std::string str;
		read(ec, std::move(section), std::move(key), str);
		if (ec)
			return;
		if (str.empty())
			value = 0;
		else
		{
			std::from_string(ec, str, value, base);
			if (ec)
				return;
		}
		ec.clear();
	}

	inline void read(
		std::string section,
		std::string key,
		std::integral auto& value,
		int base = 10
	) const
	{
		std::error_code ec;
		read(ec, std::move(section), std::move(key), value, base);
		if (ec)
			throw std::system_error(ec);
	}

	template<std::integral T>
	inline T read(
		std::string section,
		std::string key,
		int base = 10
	) const
	{
		T value;
		read(std::move(section), std::move(key), value, base);
		return T;
	}

	template<typename T>
		requires std::is_enum_v<T>
	inline void read(
		std::error_code& ec,
		std::string section,
		std::string key,
		T& value,
		int base = 10
	) const
	{
		std::underlying_type_t<T> rep;
		read(ec, std::move(section), std::move(key), rep, base);
		if (ec)
			return;
		value = T(rep);
	}

	template<typename T>
		requires std::is_enum_v<T>
	inline void read(
		std::string section,
		std::string key,
		T& value,
		int base = 10
	) const
	{
		std::error_code ec;
		read(ec, std::move(section), std::move(key), value, base);
		if (ec)
			throw std::system_error(ec);
	}

	template<typename T>
		requires std::is_enum_v<T>
	inline T read(
		std::string section,
		std::string key,
		int base = 10
	) const
	{
		T value;
		read(std::move(section), std::move(key), value, base);
		return T;
	}

	inline void read(
		std::error_code& ec,
		std::string section,
		std::string key,
		std::floating_point auto& value,
		std::chars_format fmt = std::chars_format::general
	) const
	{
		std::string str;
		read(ec, std::move(section), std::move(key), str);
		if (ec)
			return;
		if (str.empty())
			value = 0;
		else
		{
			std::from_string(ec, str, value, fmt);
			if (ec)
				return;
		}
		ec.clear();
	}

	inline void read(
		std::string section,
		std::string key,
		std::floating_point auto& value,
		std::chars_format fmt = std::chars_format::general
	) const
	{
		std::error_code ec;
		read(ec, std::move(section), std::move(key), value, fmt);
		if (ec)
			throw std::system_error(ec);
	}

	template<std::floating_point T>
	inline T read(
		std::string section,
		std::string key,
		std::chars_format fmt = std::chars_format::general
	) const
	{
		T value;
		read(std::move(section), std::move(key), value, fmt);
		return T;
	}

	void save(std::error_code& ec);

	inline void save()
	{
		std::error_code ec;
		save(ec);
		if (ec)
			throw std::system_error(ec);
	}

	void save(std::error_code& ec, std::filesystem::path path);

	inline void save(std::filesystem::path path)
	{
		std::error_code ec;
		save(ec, std::move(path));
		if (ec)
			throw std::system_error(ec);
	}

	void write(std::error_code& ec, std::string section, std::string key, std::string value);

	inline void write(std::string section, std::string key, std::string value)
	{
		std::error_code ec;
		write(ec, std::move(section), std::move(key), std::move(value));
		if (ec)
			throw std::system_error(ec);
	}

	inline void write(std::error_code& ec, std::string section, std::string key, const std::filesystem::path& path)
	{
		write(ec, std::move(section), std::move(key), path.string());
	}

	inline void write(std::string section, std::string key, const std::filesystem::path& path)
	{
		std::error_code ec;
		write(ec, std::move(section), std::move(key), path);
		if (ec)
			throw std::system_error(ec);
	}

	inline void write(
		std::error_code& ec,
		std::string section,
		std::string key,
		std::integral auto value,
		int base = 10
	)
	{
		std::string str;
		std::to_string(ec, value, str, base);
		if (ec)
			return;
		write(ec, std::move(section), std::move(key), std::move(str));
	}

	inline void write(
		std::string section,
		std::string key,
		std::integral auto value,
		int base = 10
	)
	{
		std::error_code ec;
		write(ec, std::move(section), std::move(key), value, base);
		if (ec)
			throw std::system_error(ec);
	}

	inline void write(
		std::error_code& ec,
		std::string section,
		std::string key,
		std::floating_point auto value,
		int precision = 6,
		std::chars_format fmt = std::chars_format::fixed
	)
	{
		std::string str;
		std::to_string(ec, value, str, precision, fmt);
		if (ec)
			return;
		write(ec, std::move(section), std::move(key), std::move(str));
	}

	inline void write(
		std::string section,
		std::string key,
		std::floating_point auto value,
		int precision = 6,
		std::chars_format fmt = std::chars_format::fixed
	)
	{
		std::error_code ec;
		write(ec, std::move(section), std::move(key), value, precision, fmt);
		if (ec)
			throw std::system_error(ec);
	}
};

}

#endif
