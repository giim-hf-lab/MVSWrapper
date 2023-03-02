#ifndef __UTILITIES_WINUTILS_STRINGS_HPP__
#define __UTILITIES_WINUTILS_STRINGS_HPP__

#include <string>
#include <string_view>

namespace std
{

void to_string(const wstring_view& str, string& out, bool native = true);

[[nodiscard]]
inline string to_string(const wstring_view& str, bool native = true)
{
	string out;
	to_string(str, out, native);
	return out;
}

void to_wstring(const string_view& str, wstring& out, bool native = true);

[[nodiscard]]
inline wstring to_wstring(const string_view& str, bool native = true)
{
	wstring out;
	to_wstring(str, out, native);
	return out;
}

}

#endif
