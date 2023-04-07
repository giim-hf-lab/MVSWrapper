#ifndef __UTILITIES_CAMERA_FAKE_HPP__
#define __UTILITIES_CAMERA_FAKE_HPP__

#include <cstddef>
#include <cstdint>

#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "utilities/camera/base.hpp"

namespace utilities::camera
{

class fake final : public base::device
{
	std::vector<cv::Mat> _pool;
	std::string _serial;
	base::rotation_direction _rotation;
	size_t _index;
	std::chrono::milliseconds _interval;
	std::mutex _lock;
	std::list<cv::Mat> _images;
	size_t _counter;

	std::stop_source _stop;
	std::thread _simulation;

	fake(
		const std::filesystem::path& base,
		std::string serial,
		bool colour,
		std::chrono::milliseconds interval
	);

	void _simulate(std::stop_token token);
public:
	[[nodiscard]]
	static std::vector<std::unique_ptr<fake>> find(
		const std::filesystem::path& base,
		std::vector<std::string> serials,
		bool colour,
		const std::chrono::milliseconds& interval
	);

	virtual ~fake() noexcept override;

	// base::device

	[[nodiscard]]
	inline virtual base::brand brand() const override
	{
		return base::brand::UNKNOWN;
	}

	inline virtual void close() override {}

	[[nodiscard]]
	virtual base::frame next_image(std::error_code& ec) override;

	inline virtual void open() override {}

	[[nodiscard]]
	virtual base::rotation_direction rotation() const override;

	virtual void rotation(base::rotation_direction direction) override;

	[[nodiscard]]
	virtual std::string serial() const override;

	virtual void start() override;

	virtual void stop() override;

	virtual void subscribe() override;

	inline virtual void unsubscribe() override {}
};

}

#endif
