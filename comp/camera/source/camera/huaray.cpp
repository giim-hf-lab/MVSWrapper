#include <cstddef>

#include <memory>
#include <system_error>
#include <vector>

#include <IMVApi.h>

#include "utilities/camera/huaray.hpp"

#include "./utils.hpp"

namespace utilities::camera
{

namespace
{

#define _EC_NAME "Huaray MV Camera"

struct mv_error_category final : public std::error_category
{
	[[nodiscard]]
	virtual const char *name() const noexcept override
	{
		return _EC_NAME;
	}

	[[nodiscard]]
	virtual std::string message(int condition) const override
	{
		return _UTILITIES_FORMAT_STRING(_EC_NAME" error (code {})", condition);
	}
};

static const mv_error_category _mv_error_category;

_UTILITIES_FUNCTION_TEMPLATE(F, Args, int)
inline void _wrap_mv(std::error_code& ec, F&& f, Args&&... args)
{
	if (auto ret = f(std::forward<Args>(args)...); ret != IMV_OK)
		ec.assign(ret, _mv_error_category);
	else
		ec.clear();
}

_UTILITIES_FUNCTION_TEMPLATE(F, Args, int)
inline void _wrap_mv(F&& f, Args&&... args)
{
	std::error_code ec;
	_wrap_mv(ec, std::forward<F>(f), std::forward<Args>(args)...);
	if (ec)
		throw std::system_error(ec);
}

}

void huaray::_callback(::IMV_Frame *frame, void *user)
{
	auto self = reinterpret_cast<huaray *>(user);
	const auto required_pixel_type = self->_colour ?
		::IMV_EPixelType::gvspPixelBGR8 :
		::IMV_EPixelType::gvspPixelMono8;

	cv::Mat image;
	if (const auto& info = frame->frameInfo; info.pixelFormat == required_pixel_type)
		cv::Mat(
			info.height,
			info.width,
			self->_colour ? CV_8UC3 : CV_8UC1,
			frame->pData
		).copyTo(image);
	else
	{
		image.create(info.height, info.width, self->_colour ? CV_8UC3 : CV_8UC1);

		::IMV_PixelConvertParam param {};
		param.nWidth = info.width;
		param.nHeight = info.height;
		param.ePixelFormat = info.pixelFormat;
		param.pSrcData = frame->pData;
		param.nSrcDataLen = info.size;
		param.eDstPixelFormat = required_pixel_type;
		param.pDstBuf = image.data;
		param.nDstBufSize = param.nDstDataLen = info.height * info.width * image.channels();

		_wrap_mv(::IMV_PixelConvert, self->_handle, &param);
	}

	auto guard = std::lock_guard { self->_lock };
	_utils::rotate(image, self->_images.emplace_back(), self->_rotation);
}

huaray::huaray(unsigned int index, bool colour) :
	device {},
	_handle(nullptr),
	_colour(colour),
	_rotation(base::rotation_direction::ORIGINAL),
	_lock {},
	_images {},
	_counter(0)
{
	_wrap_mv(::IMV_CreateHandle, &_handle, ::IMV_ECreateHandleMode::modeByIndex, reinterpret_cast<void *>(index));
}

[[nodiscard]]
std::vector<std::unique_ptr<huaray>> huaray::find(
	const std::vector<std::string>& serials,
	transport_layer type,
	bool colour
)
{
	::IMV_DeviceList list {};
	switch (type)
	{
		case transport_layer::USB:
			_wrap_mv(::IMV_EnumDevices, &list, ::IMV_EInterfaceType::interfaceTypeUsb3);
			break;
		case transport_layer::GIG_E:
			_wrap_mv(::IMV_EnumDevices, &list, ::IMV_EInterfaceType::interfaceTypeGige);
			break;
		default:
			return {};
	}

	std::vector<std::unique_ptr<huaray>> ret;
	if (serials.empty())
	{
		ret.reserve(list.nDevNum);
		for (size_t i = 0; i < list.nDevNum; ++i)
			ret.emplace_back(new huaray(i, colour));
	}
	else
	{
		std::unordered_map<std::string, size_t> mapping;
		for (size_t i = 0; i < list.nDevNum; ++i)
			mapping.emplace(list.pDevInfo[i].serialNumber, i);
		ret.reserve(serials.size());
		for (const auto& serial : serials)
			if (auto node = mapping.extract(serial))
				ret.emplace_back(new huaray(node.mapped(), colour));
	}
	return ret;
}

huaray::~huaray() noexcept
{
	_wrap_mv(::IMV_DestroyHandle, _handle);
	_handle = nullptr;
}

void huaray::close()
{
	_wrap_mv(::IMV_Close, _handle);
}

void huaray::open()
{
	_wrap_mv(::IMV_OpenEx, _handle, ::IMV_ECameraAccessPermission::accessPermissionControl);
}

[[nodiscard]]
base::rotation_direction huaray::rotation() const
{
	return _rotation;
}

void huaray::rotation(base::rotation_direction rotation)
{
	_rotation = rotation;
}

[[nodiscard]]
std::string huaray::serial() const
{
	::IMV_DeviceInfo device_info {};
	_wrap_mv(::IMV_GetDeviceInfo, _handle, &device_info);
	return device_info.serialNumber;
}

void huaray::start()
{
	_wrap_mv(::IMV_StartGrabbing, _handle);
}

void huaray::stop()
{
	_wrap_mv(::IMV_StopGrabbing, _handle);
}

void huaray::subscribe()
{
	_wrap_mv(::IMV_AttachGrabbing, _handle, &huaray::_callback, this);

	auto guard = std::lock_guard { _lock };
	_images.clear();
	_counter = 0;
}

void huaray::unsubscribe()
{
	_wrap_mv(::IMV_AttachGrabbing, _handle, nullptr, nullptr);
}

}
