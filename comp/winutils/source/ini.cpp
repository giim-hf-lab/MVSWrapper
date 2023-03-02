#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <windows.h>

#include "utilities/winutils/ini.hpp"
#include "utilities/winutils/strings.hpp"

namespace utilities::winutils
{

ini::ini(const std::filesystem::path& filename) : _filename(std::filesystem::absolute(filename))
{
	if (!std::filesystem::exists(_filename))
		throw std::system_error(std::make_error_code(std::errc::no_such_file_or_directory));
}

void ini::read(const std::wstring& section, const std::wstring& key, std::wstring& value)
{
	value.resize(max_value_length);
	if (auto size = ::GetPrivateProfileStringW(
		section.c_str(),
		key.c_str(),
		nullptr,
		value.data(),
		max_value_length,
		_filename.c_str()
	); !size)
		throw std::system_error(::GetLastError(), std::system_category());
	else
		value.resize(size);
}

void ini::read(const std::wstring& section, const std::wstring& key, std::string& value)
{
	std::wstring buffer;
	read(section, key, buffer);
	std::to_string(buffer, value, false);
}

void ini::write(const std::wstring& section, const std::wstring& key, const std::wstring& value)
{
	if (value.size() > max_value_length)
		throw std::system_error(std::make_error_code(std::errc::value_too_large));
	if (!::WritePrivateProfileStringW(section.c_str(), key.c_str(), value.c_str(), _filename.c_str()))
		throw std::system_error(::GetLastError(), std::system_category());
}

void ini::write(const std::wstring& section, const std::wstring& key, const std::string_view& value)
{
	write(section, key, std::to_wstring(value, false));
}

}
