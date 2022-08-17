#ifndef __WRAPPER_SOCKET_HPP__
#define __WRAPPER_SOCKET_HPP__

#include <cerrno>
#include <cstddef>
#include <cstdint>

#include <ranges>
#include <string>
#include <system_error>
#include <type_traits>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace wrapper::net
{

enum class SOCKET_DOMAIN
{
	IPv4 = AF_INET,
	IPv6 = AF_INET6
};

class client final
{
	[[using gnu : always_inline]]
	inline static void _check(int ret)
	{
		if (ret < 0)
		[[unlikely]]
			throw std::system_error(errno, std::system_category(), "socket operation failed");
	}

	[[using gnu : always_inline]]
	inline static void _check_pton(int ret)
	{
		switch (ret)
		{
			case 1:
			[[likely]]
				return;
			case 0:
			[[unlikely]]
				throw std::system_error(EINVAL, std::system_category(), "invalid address range");
			case -1:
			[[unlikely]]
				throw std::system_error(EAFNOSUPPORT, std::generic_category(), "invalid address");
			default:
			[[unlikely]]
				throw std::logic_error("unexpected return value from inet_pton");
		}
	}

	SOCKET_DOMAIN _domain;
	int _fd;
public:
	client(SOCKET_DOMAIN domain, const std::string & address, uint16_t port) :
		_domain(domain),
		_fd(::socket(static_cast<int>(domain), SOCK_STREAM, IPPROTO_TCP))
	{
		_check(_fd);

		::sockaddr_storage addr_in {};
		switch (domain)
		{
			case SOCKET_DOMAIN::IPv4:
			{
				auto & addr_in_4 = reinterpret_cast<::sockaddr_in &>(addr_in);
				addr_in_4.sin_family = static_cast<int>(domain);
				addr_in_4.sin_port = ::htons(port);
				_check_pton(::inet_pton(static_cast<int>(domain), address.c_str(), &addr_in_4.sin_addr));
				return;
			}
			case SOCKET_DOMAIN::IPv6:
			{
				auto & addr_in_6 = reinterpret_cast<::sockaddr_in6 &>(addr_in);
				addr_in_6.sin6_family = static_cast<int>(domain);
				addr_in_6.sin6_port = ::htons(port);
				_check_pton(::inet_pton(static_cast<int>(domain), address.c_str(), &addr_in_6.sin6_addr));
				return;
			}
			default:
				throw std::system_error(EINVAL, std::generic_category(), "invalid socket domain");
		}
		_check(::connect(_fd, reinterpret_cast<::sockaddr *>(&addr_in), sizeof(addr_in)));
	}

	client(const client &) = delete;
	client(client &&) noexcept = default;

	client & operator=(const client &) = delete;
	client & operator=(client &&) = delete;

	void keep_alive(bool enable) &
	{
		int opt = enable;
		_check(::setsockopt(_fd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt)));
	}

	template<std::ranges::contiguous_range C>
	size_t recv(C & container, int flags = 0) &
	{
		using T = std::ranges::range_value_t<C>;
		static_assert(std::is_trivial_v<T>, "container element must be trivial");

		auto ret = ::recv(_fd, container.data(), container.size() * sizeof(T), flags);
		_check(ret);
		return ret;
	}

	template<std::ranges::contiguous_range C>
	size_t send(const C & container, int flags = 0) &
	{
		using T = std::ranges::range_value_t<C>;
		static_assert(std::is_trivial_v<T>, "container element must be trivial");

		auto ret = ::send(_fd, container.data(), container.size() * sizeof(T), flags);
		_check(ret);
		return ret;
	}
};

}

#endif
