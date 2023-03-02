#include <string>
#include <string_view>
#include <system_error>

#include <windows.h>

#include "utilities/winutils/strings.hpp"

namespace std
{

void to_string(const wstring_view& str, string& out, bool native)
{
	out.resize(str.size() * 4);
	if (auto size = ::WideCharToMultiByte(
		native ? CP_ACP : CP_UTF8,
		0,
		str.data(),
		str.size(),
		out.data(),
		out.size(),
		nullptr,
		nullptr
	); !size)
		throw system_error(::GetLastError(), system_category());
	else
		out.resize(size);
}

void to_wstring(const string_view& str, wstring& out, bool native)
{
	out.resize(str.size() * 2);
	if (auto size = ::MultiByteToWideChar(
		native ? CP_ACP : CP_UTF8,
		0,
		str.data(),
		str.size(),
		out.data(),
		out.size()
	); !size)
		throw system_error(::GetLastError(), system_category());
	else
		out.resize(size);
}

}
