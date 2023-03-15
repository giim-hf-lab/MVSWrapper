#define MINI_CASE_SENSITIVE

#include <filesystem>
#include <system_error>
#include <utility>

#include <mini/ini.h>

#include "utilities/ini.hpp"

namespace utilities
{

ini::ini() noexcept : _path {}, _data {}, _modified(false) {}

ini::ini(std::filesystem::path path) : ini()
{
	if (std::filesystem::is_regular_file(path))
		load(std::move(path));
	else
		_path = std::move(path);
}

[[nodiscard]]
bool ini::created() const noexcept
{
	return !_path.empty();
}

void ini::load(std::error_code& ec, std::filesystem::path path)
{
	if (!std::filesystem::exists(path))
	{
		ec = std::make_error_code(std::errc::no_such_file_or_directory);
		return;
	}

	if (!(mINI::INIReader { path.string() } >> _data))
	{
		ec = std::make_error_code(std::errc::invalid_argument);
		return;
	}

	_path = std::move(path);
	_modified = false;
	ec.clear();
}

[[nodiscard]]
bool ini::modified() const noexcept
{
	return _modified;
}

void ini::read(std::error_code& ec, std::string section, std::string key, std::string& value) const
{
	value = _data.get(std::move(section)).get(std::move(key));
	ec.clear();
}

void ini::save(std::error_code& ec)
{
	if (_path.empty())
	{
		ec = std::make_error_code(std::errc::no_such_file_or_directory);
		return;
	}

	if (!(mINI::INIWriter { _path.string() } << _data))
	{
		ec = std::make_error_code(std::errc::invalid_argument);
		return;
	}

	_modified = false;
	ec.clear();
}

void ini::save(std::error_code& ec, std::filesystem::path path)
{
	std::filesystem::create_directories(path.parent_path(), ec);
	if (ec)
		return;
	_path = std::move(path);
	save(ec);
}

void ini::write(std::error_code& ec, std::string section, std::string key, std::string value)
{
	_data[std::move(section)].set(std::move(key), std::move(value));
	_modified = true;
	ec.clear();
}

}
