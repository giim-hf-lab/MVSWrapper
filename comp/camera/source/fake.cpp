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

void fake::_simulate(std::stop_token token)
{
	while (!token.stop_requested())
	{
		auto now = std::chrono::steady_clock::now();
		const auto selection = _selection_distribution(_selection_generator);
		const std::chrono::milliseconds required_elapsed { _base_interval + _offset_distribution(_time_generator) };
		while (std::chrono::steady_clock::now() - now < required_elapsed)
		{
			std::this_thread::sleep_for(1ms);
			if (token.stop_requested())
				return;
		}

		const auto& image = _pool[selection];
		auto guard = std::lock_guard { _lock };
		_utils::rotate(image, _images.emplace_back(), _rotation);
	}
}

namespace
{

static const std::unordered_set<std::filesystem::path> _accepted_extension {
	".bmp",
	".jpg",
	".png"
};

[[nodiscard]]
std::vector<cv::Mat> _load_images(const std::filesystem::path& directory, bool colour)
{
	const auto read_mode = colour ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
	std::vector<cv::Mat> images;
	for (const auto& entry : std::filesystem::directory_iterator(directory))
	{
		if (entry.is_regular_file())
			if (const auto& path = entry.path(); _accepted_extension.count(path.extension()))
				images.push_back(cv::imread(path.string(), read_mode));
	}
	return images;
}

}

[[nodiscard]]
std::vector<std::unique_ptr<fake>> fake::find(
	const std::filesystem::path& base,
	std::vector<std::string> serials,
	bool colour,
	size_t base_interval,
	int64_t offset_range
)
{
	std::vector<std::unique_ptr<fake>> ret;
	ret.reserve(serials.size());
	for (auto& serial : serials)
		ret.emplace_back(new fake(base, std::move(serial), colour, base_interval, offset_range));
	return ret;
}

fake::fake(
	const std::filesystem::path& base,
	std::string serial,
	bool colour,
	size_t base_interval,
	int64_t offset_range
) :
	device {},
	_pool { _load_images(base / serial, colour) },
	_serial { std::move(serial) },
	_rotation(base::rotation_direction::ORIGINAL),
	_base_interval(base_interval),
	_seed_generator {},
	_time_generator { _seed_generator() },
	_selection_generator { _seed_generator() },
	_offset_distribution { -offset_range, offset_range },
	_selection_distribution { 0, _pool.size() - 1 },
	_lock {},
	_images {},
	_stop {},
	_simulation {}
{}

fake::~fake() noexcept
{
	stop();
}

[[nodiscard]]
bool fake::next_image(std::error_code& ec, cv::Mat& image)
{
	auto guard = std::lock_guard { _lock };
	if (_images.empty())
		return false;
	image = std::move(_images.front());
	_images.pop_front();
	return true;
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

void fake::start(bool latest_only)
{
	auto guard = std::lock_guard { _lock };
	_images.clear();

	_stop = {};
	_simulation = std::thread { &fake::_simulate, this, _stop.get_token() };
}

void fake::stop()
{
	_stop.request_stop();
	if (_simulation.joinable())
		_simulation.join();
}

}
