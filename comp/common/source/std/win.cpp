#include <cstddef>

#include <string>
#include <string_view>
#include <system_error>

#include <windows.h>

#include "utilities/std/win.hpp"

namespace std
{

void from_string(error_code& ec, const string_view& str, wstring& out, bool native, bool low_memory) noexcept
{
	const auto code_page = native ? CP_ACP : CP_UTF8;

	size_t estimated;
	if (low_memory)
	{
		estimated = ::MultiByteToWideChar(
			code_page,
			0,
			str.data(),
			str.size(),
			nullptr,
			0
		);
		if (!estimated)
		{
			ec.assign(::GetLastError(), system_category());
			out.clear();
			return;
		}
	}
	else
		estimated = str.size() * 2;

	out.resize(estimated);
	if (auto size = ::MultiByteToWideChar(
		code_page,
		0,
		str.data(),
		str.size(),
		out.data(),
		estimated
	))
	{
		ec.clear();
		out.resize(size);
	}
	else
	{
		ec.assign(::GetLastError(), system_category());
		out.clear();
	}
}

void to_string(error_code& ec, const wstring_view& str, string& out, bool native, bool low_memory) noexcept
{
	const auto code_page = native ? CP_ACP : CP_UTF8;

	size_t estimated;
	if (low_memory)
	{
		estimated = ::WideCharToMultiByte(
			code_page,
			0,
			str.data(),
			str.size(),
			nullptr,
			0,
			nullptr,
			nullptr
		);
		if (!estimated)
		{
			ec.assign(::GetLastError(), system_category());
			out.clear();
			return;
		}
	}
	else
		estimated = str.size() * 4;

	out.resize(estimated);
	if (auto size = ::WideCharToMultiByte(
		code_page,
		0,
		str.data(),
		str.size(),
		out.data(),
		estimated,
		nullptr,
		nullptr
	))
	{
		ec.clear();
		out.resize(size);
	}
	else
	{
		ec.assign(::GetLastError(), system_category());
		out.clear();
	}
}

}
