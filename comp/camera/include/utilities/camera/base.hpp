#ifndef __UTILITIES_CAMERA_BASE_HPP__
#define __UTILITIES_CAMERA_BASE_HPP__

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
	MVS
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
	virtual bool next_image(std::error_code& ec, cv::Mat& image) = 0;

	[[nodiscard]]
	inline virtual bool next_image(cv::Mat& image) final
	{
		std::error_code ec;
		auto ret = next_image(ec, image);
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

	virtual void start(bool latest_only) = 0;

	virtual void stop() = 0;

	virtual void subscribe() = 0;

	virtual void unsubscribe() = 0;
};

}

#endif
