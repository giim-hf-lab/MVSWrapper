#ifndef __UTILITIES_CAMERA_BASE_HPP__
#define __UTILITIES_CAMERA_BASE_HPP__

#include <cstddef>

#include <string>
#include <system_error>
#include <utility>

#include <opencv2/core.hpp>

namespace utilities::camera::base
{

enum class brand
{
	UNKNOWN,
	BASLER,
	HIKVISION,
	HUARAY
};

struct frame final
{
	size_t id;
	cv::Mat content;

	inline frame() noexcept : id(0), content {} {}

	inline frame(size_t id, cv::Mat content) noexcept : id(id), content(std::move(content)) {}

	frame(const frame&) = delete;

	inline frame(frame&&) noexcept = default;

	[[nodiscard]]
	inline operator bool() const noexcept { return id; }
};

enum class rotation_direction
{
	ORIGINAL,
	CLOCKWISE_90,
	ANY_180,
	COUNTER_CLOCKWISE_90
};

struct device
{
	inline device() noexcept = default;

	device(const device&) = delete;

	inline device(device&&) noexcept = default;

	inline virtual ~device() noexcept = default;

	[[nodiscard]]
	virtual brand brand() const = 0;

	virtual void close() = 0;

	[[nodiscard]]
	virtual frame next_image(std::error_code& ec) = 0;

	[[nodiscard]]
	inline virtual frame next_image() final
	{
		std::error_code ec;
		auto ret = next_image(ec);
		if (ec)
			throw std::system_error(ec);
		return ret;
	}

	virtual void open() = 0;

	[[nodiscard]]
	virtual rotation_direction rotation() const = 0;

	virtual void rotation(rotation_direction rotation) = 0;

	[[nodiscard]]
	virtual std::string serial() const = 0;

	virtual void start() = 0;

	virtual void stop() = 0;

	virtual void subscribe() = 0;

	virtual void unsubscribe() = 0;
};

}

#endif
