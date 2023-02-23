#pragma once

#include <cstddef>

#include <filesystem>
#include <type_traits>
#include <utility>
#include <vector>

#include <mio/mmap.hpp>

#if __cplusplus < 202002L

#define PARAMETER_PACK_TEMPLATE(N, T) template<typename... N, typename = std::enable_if_t<(T<N> && ...)>>

#else

#define PARAMETER_PACK_TEMPLATE(N, T) template<typename... N> requires (T<N> && ...)

#endif

namespace utilities
{

class filecache final
{
	std::filesystem::path _path;
	mio::mmap_sink _mmap;
	std::vector<size_t> _block_sizes;
	size_t _expansion_lines;

	template<typename... Ts, std::size_t... Is>
	[[nodiscard]]
	inline bool _read(size_t line, std::index_sequence<Is...>, Ts&... ts)
	{
		return (_read(line, Is, ts) && ...);
	}

	template<typename... Ts, std::size_t... Is>
	[[nodiscard]]
	inline bool _write(size_t line, std::index_sequence<Is...>, const Ts&... ts)
	{
		return (_write(line, Is, ts) && ...);
	}
public:
	PARAMETER_PACK_TEMPLATE(Ts, std::is_trivial_v)
	[[nodiscard]]
	inline static filecache create(std::filesystem::path path, size_t expansion_lines = 0)
	{
		return filecache(std::move(path), { sizeof(Ts)... }, expansion_lines);
	}

	filecache(std::filesystem::path path, size_t expansion_lines = 0);

	filecache(std::filesystem::path path, std::vector<size_t> block_sizes, size_t expansion_lines = 0);

	PARAMETER_PACK_TEMPLATE(Ts, std::is_trivial_v)
	[[nodiscard]]
	inline bool read(size_t line, Ts&... ts)
	{
		return sizeof...(Ts) == _block_sizes.size() && _read(line, std::index_sequence_for<Ts...> {}, ts...);
	}

	PARAMETER_PACK_TEMPLATE(Ts, std::is_trivial_v)
	[[nodiscard]]
	inline bool write(size_t line, const Ts&... ts)
	{
		return sizeof...(Ts) == _block_sizes.size() && _write(line, std::index_sequence_for<Ts...> {}, ts...);
	}
};

}
