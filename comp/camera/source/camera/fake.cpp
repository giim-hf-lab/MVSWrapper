#include <cmath>
#include <cstddef>
#include <cstdint>

#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "utilities/camera/base.hpp"
#include "utilities/camera/fake.hpp"

#include "./utils.hpp"

using std::chrono_literals::operator""ms;

namespace utilities::camera
{

namespace
{

static const std::unordered_set<std::string> _accepted_extension {
	".bmp",
	".jpg",
	".png"
};

[[nodiscard]]
static std::vector<cv::Mat> _load_images(const std::filesystem::path& directory, bool colour)
{
	const auto read_mode = colour ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
	std::vector<cv::Mat> images;
	for (const auto& entry : std::filesystem::directory_iterator(directory))
	{
		if (entry.is_regular_file())
			if (const auto& path = entry.path(); _accepted_extension.count(path.extension().string()))
				images.push_back(cv::imread(path.string(), read_mode));
	}
	return images;
}

}

fake::fake(
	const std::filesystem::path& base,
	std::string serial,
	bool colour,
	std::chrono::milliseconds interval
) :
	device {},
	_pool { _load_images(base / serial, colour) },
	_serial { std::move(serial) },
	_rotation(base::rotation_direction::ORIGINAL),
	_index(0),
	_interval(std::move(interval)),
	_lock {},
	_images {},
	_counter(0),
	_stop {},
	_simulation {}
{}

void fake::_simulate(std::stop_token token)
{
	if (_pool.empty())
		return;

	while (!token.stop_requested())
	{
		auto now = std::chrono::steady_clock::now();
		while (std::chrono::steady_clock::now() - now < _interval)
		{
			std::this_thread::sleep_for(1ms);
			if (token.stop_requested())
				return;
		}

		auto guard = std::lock_guard { _lock };
		_utils::rotate(_pool[_index++], _images.emplace_back(), _rotation);
		if (_index == _pool.size())
			_index = 0;
	}
}

[[nodiscard]]
std::vector<std::unique_ptr<fake>> fake::find(
	const std::filesystem::path& base,
	std::vector<std::string> serials,
	bool colour,
	const std::chrono::milliseconds& interval
)
{
	std::vector<std::unique_ptr<fake>> ret;
	ret.reserve(serials.size());
	for (auto& serial : serials)
		if (std::filesystem::exists(base / serial))
			ret.emplace_back(new fake(base, std::move(serial), colour, interval));
	return ret;
}

fake::~fake() noexcept
{
	stop();
}

[[nodiscard]]
base::frame fake::next_image(std::error_code& ec)
{
	auto guard = std::lock_guard { _lock };
	if (_images.empty())
		return {};

	base::frame ret { ++_counter, std::move(_images.front()) };
	_images.pop_front();
	return ret;
}

[[nodiscard]]
base::rotation_direction fake::rotation() const
{
	return _rotation;
}

void fake::rotation(base::rotation_direction rotation)
{
	_rotation = rotation;
}

[[nodiscard]]
std::string fake::serial() const
{
	return _serial;
}

void fake::start()
{
	_stop = {};
	_simulation = std::thread { &fake::_simulate, this, _stop.get_token() };
}

void fake::stop()
{
	_stop.request_stop();
	if (_simulation.joinable())
		_simulation.join();
}

void fake::subscribe()
{
	auto guard = std::lock_guard { _lock };
	_images.clear();
	_counter = 0;
}

}
