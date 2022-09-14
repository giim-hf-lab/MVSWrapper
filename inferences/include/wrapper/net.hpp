#ifndef INFERENCES_WRAPPER_SOCKET_HPP
#define INFERENCES_WRAPPER_SOCKET_HPP

#include <cstddef>
#include <cstdint>

#include <concepts>
#include <ranges>
#include <string>

#include <sys/socket.h>

namespace inferences::wrapper::net
{

enum class SOCKET_DOMAIN
{
	IPv4 = AF_INET,
	IPv6 = AF_INET6
};

class client final
{
	SOCKET_DOMAIN _domain;
	int _fd;

	[[nodiscard]]
	size_t _raw_recv(void *data, size_t size, int flags) &;

	[[nodiscard]]
	size_t _raw_send(const void *data, size_t size, int flags) &;
public:
	client(SOCKET_DOMAIN domain, const std::string& address, uint16_t port);

	~client() noexcept;

	client(const client&) = delete;

	client(client&&) noexcept;

	client& operator=(const client&) = delete;

	client& operator=(client&&) = delete;

	void keep_alive(bool enable) &;

	template<std::integral T>
	size_t recv(T *data, size_t count) &
	{
		return _raw_recv(data, count * sizeof(T), 0);
	}

	template<std::ranges::contiguous_range C>
	inline size_t recv(C& data) &
	{
		return recv(std::ranges::data(data), std::ranges::size(data), 0);
	}

	template<std::integral T>
	size_t send(const T *data, size_t count) &
	{
		return _raw_send(data, count * sizeof(T), 0);
	}

	template<std::ranges::contiguous_range C>
	inline size_t send(const C& data) &
	{
		return send(std::ranges::data(data), std::ranges::size(data), 0);
	}
};

}

#endif
