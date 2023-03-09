#include <cstddef>

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <opencv2/core.hpp>
#include <pylon/_BaslerUniversalCameraParams.h>
#include <pylon/BaslerUniversalInstantCamera.h>
#include <pylon/PylonIncludes.h>

#include "utilities/camera/base.hpp"
#include "utilities/camera/basler.hpp"

namespace utilities::camera
{

basler::initialiser::initialiser()
{
	Pylon::PylonInitialize();
}

basler::initialiser::~initialiser() noexcept
{
	Pylon::PylonTerminate();
}

basler::image_listener::image_listener(bool colour) :
	CBaslerUniversalImageEventHandler {},
	_converter {},
	_colour(colour),
	_images {},
	_lock {}
{
	_converter.Initialize(_colour ? Pylon::PixelType_BGR8packed : Pylon::PixelType_Mono8);
}

basler::image_listener::~image_listener() noexcept
{
	_converter.Uninitialize();
}

bool basler::image_listener::next(std::error_code& ec, cv::Mat& image)
{
	auto guard = std::lock_guard(_lock);
	if (_images.empty())
		return false;
	image = std::move(_images.front());
	_images.pop_front();
	return true;
}

void basler::image_listener::OnImageGrabbed(
	Pylon::CBaslerUniversalInstantCamera& camera,
	const Pylon::CBaslerUniversalGrabResultPtr& grabResult
)
{
	Pylon::IImage& source_image = grabResult;
	cv::Mat image;
	if (_converter.ImageHasDestinationFormat(source_image))
	{
		size_t stride;
		if (!source_image.GetStride(stride))
			throw std::logic_error("failed to get the stride of BGR/Mono image");
		cv::Mat(
			source_image.GetHeight(),
			source_image.GetWidth(),
			_colour ? CV_8UC3 : CV_8UC1,
			source_image.GetBuffer(),
			stride
		).copyTo(image);
	}
	else
	{
		image.create(source_image.GetHeight(), source_image.GetWidth(), _colour ? CV_8UC3 : CV_8UC1);
		_converter.Convert(image.data, image.total() * image.channels(), source_image);
	}

	auto guard = std::lock_guard(_lock);
	_images.push_back(std::move(image));
}

const basler::initialiser basler::_global_guard;

basler::basler(const Pylon::CDeviceInfo& device_info, bool colour) :
	device {},
	_instance { Pylon::CTlFactory::GetInstance().CreateDevice(device_info), Pylon::Cleanup_Delete },
	_listener { colour }
{}

[[nodiscard]]
std::vector<std::unique_ptr<basler>> basler::find(
	const std::vector<std::string_view>& serials,
	transport_layer type,
	bool colour
)
{
	Pylon::CDeviceInfo reference;
	switch (type)
	{
		case transport_layer::USB:
			reference.SetTLType(Pylon::TLType::TLTypeUSB);
			break;
		case transport_layer::GIG_E:
			reference.SetTLType(Pylon::TLType::TLTypeGigE);
			break;
	}

	Pylon::DeviceInfoList filter;
	filter.assign((std::max)(serials.size(), size_t(1)), reference);
	for (size_t i = 0; i < serials.size(); ++i)
	{
		const auto& serial = serials[i];
		filter[i].SetSerialNumber({ serial.data(), serial.size() });
	}

	Pylon::DeviceInfoList devices;
	Pylon::CTlFactory::GetInstance().EnumerateDevices(devices, filter, true);

	std::vector<std::unique_ptr<basler>> ret;
	ret.reserve(devices.size());
	for (auto& device_info : devices)
		ret.emplace_back(new basler(device_info, colour));
	return ret;
}

void basler::close()
{
	_instance.Close();
}

[[nodiscard]]
bool basler::next_image(std::error_code& ec, cv::Mat& image)
{
	return _listener.next(ec, image);
}

void basler::open()
{
	_instance.Open();
}

std::string basler::serial() const
{
	const auto& serial = _instance.GetDeviceInfo().GetSerialNumber();
	return { serial.c_str(), serial.size() };
}

void basler::start(bool latest_only)
{
	_instance.StartGrabbing(
		latest_only ?
			Pylon::GrabStrategy_LatestImageOnly :
			Pylon::GrabStrategy_OneByOne,
		Pylon::GrabLoop_ProvidedByInstantCamera
	);
}

void basler::stop()
{
	_instance.StopGrabbing();
}

void basler::subscribe()
{
	_instance.RegisterImageEventHandler(
		&_listener,
		Pylon::RegistrationMode_ReplaceAll,
		Pylon::Cleanup_None
	);
}

void basler::unsubscribe()
{
	_instance.DeregisterImageEventHandler(&_listener);
}

[[nodiscard]]
bool basler::set_exposure_time(const std::chrono::duration<double, std::micro>& time)
{
	return _instance.ExposureAuto.TrySetValue(Basler_UniversalCameraParams::ExposureAuto_Off) &&
		_instance.ExposureTime.TrySetValue(time.count()) &&
		_instance.ExposureMode.TrySetValue(Basler_UniversalCameraParams::ExposureMode_Timed);
}

[[nodiscard]]
bool basler::set_gain(double gain)
{
	return _instance.GainAuto.TrySetValue(Basler_UniversalCameraParams::GainAuto_Off) &&
		_instance.Gain.TrySetValue(gain);
}

namespace
{

static constexpr auto _universal_lines = std::array {
	Basler_UniversalCameraParams::LineSelector_Line1,
	Basler_UniversalCameraParams::LineSelector_Line2,
	Basler_UniversalCameraParams::LineSelector_Line3,
	Basler_UniversalCameraParams::LineSelector_Line4
};

}

[[nodiscard]]
bool basler::set_line_debouncer_time(size_t line, const std::chrono::duration<double, std::micro>& time)
{
	if (line >= _universal_lines.size())
		return false;

	return _instance.LineSelector.TrySetValue(_universal_lines[line]) &&
		_instance.LineDebouncerTime.TrySetValue(time.count());
}

namespace
{

static constexpr auto _trigger_lines = std::array {
	Basler_UniversalCameraParams::TriggerSource_Line1,
	Basler_UniversalCameraParams::TriggerSource_Line2,
	Basler_UniversalCameraParams::TriggerSource_Line3
};

}

[[nodiscard]]
bool basler::set_manual_trigger_line_source(size_t line, const std::chrono::duration<double, std::micro>& delay)
{
	if (line >= _trigger_lines.size())
		return false;

	return _instance.TriggerSelector.TrySetValue(Basler_UniversalCameraParams::TriggerSelector_FrameBurstStart) &&
		_instance.TriggerSource.TrySetValue(_trigger_lines[line]) &&
		_instance.TriggerActivation.TrySetValue(Basler_UniversalCameraParams::TriggerActivation_RisingEdge) &&
		_instance.TriggerDelay.TrySetValue(delay.count()) &&
		_instance.TriggerMode.TrySetValue(Basler_UniversalCameraParams::TriggerMode_On);
}

}
