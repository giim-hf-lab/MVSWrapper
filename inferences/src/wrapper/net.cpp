#include <cerrno>
#include <cstddef>
#include <cstdint>

#include <stdexcept>
#include <string>
#include <system_error>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "wrapper/net.hpp"

#include "./utils/unix.hpp"

namespace
{

template<typename... Args>
inline static void _inet_pton(Args&&... args)
{
	switch (::inet_pton(std::forward<Args>(args)...))
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

}

namespace inferences::wrapper::net
{

[[nodiscard]]
size_t client::_raw_recv(void *data, size_t size, int flags) &
{
	return utils::_wrap_unix(::recv, _fd, data, size, flags);
}

[[nodiscard]]
size_t client::_raw_send(const void *data, size_t size, int flags) &
{
	return utils::_wrap_unix(::send, _fd, data, size, flags);
}

client::client(SOCKET_DOMAIN domain, const std::string& address, uint16_t port) :
	_domain(domain),
	_fd(utils::_wrap_unix(::socket, static_cast<int>(domain), SOCK_STREAM, IPPROTO_TCP))
{
	::sockaddr_storage addr_in {};
	switch (domain)
	{
		case SOCKET_DOMAIN::IPv4:
		{
			auto& addr_in_4 = reinterpret_cast<::sockaddr_in &>(addr_in);
			addr_in_4.sin_family = static_cast<int>(domain);
			addr_in_4.sin_port = ::htons(port);
			_inet_pton(static_cast<int>(domain), address.c_str(), &addr_in_4.sin_addr);
			return;
		}
		case SOCKET_DOMAIN::IPv6:
		{
			auto& addr_in_6 = reinterpret_cast<::sockaddr_in6 &>(addr_in);
			addr_in_6.sin6_family = static_cast<int>(domain);
			addr_in_6.sin6_port = ::htons(port);
			_inet_pton(static_cast<int>(domain), address.c_str(), &addr_in_6.sin6_addr);
			return;
		}
		default:
			throw std::system_error(EINVAL, std::generic_category(), "invalid socket domain");
	}
	utils::_wrap_unix(::connect, _fd, reinterpret_cast<::sockaddr *>(&addr_in), sizeof(addr_in));
}

client::~client() noexcept
{
	utils::_wrap_unix(::close, _fd);
}

client::client(client&&) noexcept = default;

void client::keep_alive(bool enable) &
{
	int opt = enable;
	utils::_wrap_unix(::setsockopt, _fd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt));
}

}
