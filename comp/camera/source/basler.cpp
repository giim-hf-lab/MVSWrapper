#include <cstddef>

#include <memory>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

#include <opencv2/core.hpp>
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
	CBaslerUniversalImageEventHandler(),
	_converter(),
	_colour(colour),
	_images(),
	_lock()
{
	_converter.Initialize(_colour ? Pylon::EPixelType::PixelType_BGR8packed : Pylon::EPixelType::PixelType_Mono8);
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
		image.create(source_image.GetHeight(), source_image.GetWidth(), _colour ? CV_8UC3 : CV_8UC1);
		_converter.Convert(image.data, image.total() * image.channels(), source_image);
	}
	else
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

	auto guard = std::lock_guard(_lock);
	_images.push_back(std::move(image));
}

basler::basler(const Pylon::CDeviceInfo& device_info, bool colour) :
	device(),
	_instance(Pylon::CTlFactory::GetInstance().CreateDevice(device_info), Pylon::ECleanup::Cleanup_Delete),
	_listener(colour)
{}

[[nodiscard]]
std::vector<std::unique_ptr<base::device>> basler::find(
	const std::vector<std::string_view>& serials,
	transport_layer type,
	bool colour
)
{
	Pylon::DeviceInfoList filter(serials.size());
	for (size_t i = 0; i < serials.size(); ++i)
	{
		const auto& serial = serials[i];
		auto& device_info = filter[i];
		device_info.SetSerialNumber({ serial.data(), serial.size() });
		switch (type)
		{
			case transport_layer::USB:
				device_info.SetTLType(Pylon::TLType::TLTypeUSB);
				break;
			case transport_layer::GIG_E:
				device_info.SetTLType(Pylon::TLType::TLTypeGigE);
				break;
		}
	}

	Pylon::DeviceInfoList devices;
	devices.reserve(serials.size());
	Pylon::CTlFactory::GetInstance().EnumerateDevices(devices, filter, true);

	std::vector<std::unique_ptr<base::device>> ret;
	ret.reserve(devices.size());
	for (auto& device_info : devices)
		ret.emplace_back(new basler(device_info, colour));
	return ret;
}

void basler::close()
{
	_instance.Close();
}

void basler::open()
{
	_instance.Open();
}

void basler::start(bool latest_only)
{
	_instance.StartGrabbing(
		latest_only ?
			Pylon::EGrabStrategy::GrabStrategy_LatestImageOnly :
			Pylon::EGrabStrategy::GrabStrategy_OneByOne
	);
}

void basler::stop()
{
	_instance.StopGrabbing();
}

void basler::subscribe(bool exclusive)
{
	_instance.RegisterImageEventHandler(
		&_listener,
		exclusive ?
			Pylon::ERegistrationMode::RegistrationMode_ReplaceAll :
			Pylon::ERegistrationMode::RegistrationMode_Append,
		Pylon::ECleanup::Cleanup_None
	);
}

void basler::unsubscribe()
{
	_instance.DeregisterImageEventHandler(&_listener);
}

[[nodiscard]]
bool basler::next_image(std::error_code& ec, cv::Mat& image)
{
	return _listener.next(ec, image);
}

}
