#include <chrono>
#include <memory>
#include <vector>

#include "utilities/camera/base.hpp"
#include "utilities/camera/basler.hpp"
#include "utilities/camera/mvs.hpp"

using std::chrono_literals::operator""ms;
using std::chrono_literals::operator""us;

int main()
{
	auto basler_cameras = utilities::camera::basler::find(
		{},
		utilities::camera::basler::transport_layer::GIG_E,
		false
	);
	for (auto& camera : basler_cameras)
	{
		camera->open();
		if (
			!camera->set_exposure_time(100us) ||
			!camera->set_gain(10.0) ||
			!camera->set_line_debouncer_time(0, 20ms) ||
			!camera->set_manual_trigger_line_source(0, 0us)
		)
			return 1;
	}

	auto mvs_cameras = utilities::camera::mvs::find(
		{},
		utilities::camera::mvs::transport_layer::GIG_E,
		false
	);
	for (auto& camera : mvs_cameras)
	{
		camera->open();
		if (
			!camera->set_exposure_time(100us) ||
			!camera->set_gain(10.0) ||
			!camera->set_line_debouncer_time(0, 20ms) ||
			!camera->set_manual_trigger_line_source(0, 0us)
		)
			return 2;
	}

	std::vector<utilities::camera::base::device *> all_cameras;
	all_cameras.reserve(basler_cameras.size());
	for (auto& camera : basler_cameras)
		all_cameras.push_back(camera.get());
	for (auto& camera : mvs_cameras)
		all_cameras.push_back(camera.get());

	for (auto& camera : all_cameras)
		camera->subscribe();

	return 0;
}
