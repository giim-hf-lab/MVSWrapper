#ifndef __UTILITIES_WINUTILS_INI_HPP__
#define __UTILITIES_WINUTILS_INI_HPP__

#include <cstddef>

#include <charconv>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

namespace utilities::winutils
{

class ini final
{
	std::filesystem::path _filename;
public:
	static constexpr size_t max_value_length = 1024;

	ini(const std::filesystem::path& filename);

	void read(const std::wstring& section, const std::wstring& key, std::wstring& value);

	void read(const std::wstring& section, const std::wstring& key, std::string& value);

	template<typename T>
	void read(const std::wstring& section, const std::wstring& key, T& value)
	{
		std::string buffer;
		read(section, key, buffer);
		auto sp = buffer.c_str(), ep = sp + buffer.size();
		if (auto result = std::from_chars(sp, ep, value); result.ec)
			throw std::system_error(std::make_error_code(result.ec));
	}

	void write(const std::wstring& section, const std::wstring& key, const std::wstring& value);

	void write(const std::wstring& section, const std::wstring& key, const std::string_view& value);

	template<typename T>
	void write(const std::wstring& section, const std::wstring& key, const T& value)
	{
		std::string buffer(max_value_length / 4, 0);
		auto sp = buffer.data(), ep = sp + buffer.size();
		if (auto result = std::to_chars(sp, ep, value); result.ec)
			throw std::system_error(std::make_error_code(result.ec));
		else
			buffer.resize(result.ptr - sp);
		write(section, key, buffer);
	}
};

}

#endif
