#include <cstddef>

#include <charconv>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <MvCameraControl.h>

#include "utilities/camera/base.hpp"
#include "utilities/camera/mvs.hpp"

namespace utilities::camera
{

namespace
{

struct mvs_error_category final : public std::error_category
{
	virtual const char *name() const noexcept override
	{
		return "HikVision MV Camera";
	}

	virtual std::string message(int condition) const override
	{
		static constexpr size_t code_width = sizeof(int) * 2;
		char code_string[code_width];
		auto r = std::to_chars(code_string, code_string + code_width, condition, 16);
		if (std::error_code ec = std::make_error_code(r.ec); ec)
			throw std::system_error(ec);

		size_t actual_code_width = r.ptr - code_string, padding_width = code_width - actual_code_width;
		static constexpr std::string_view prefix = "HikVision MV Camera error (code 0x", suffix = ")";
		static constexpr size_t total_width = prefix.size() + code_width + suffix.size() + 1;

		std::string ret;
		ret.reserve(total_width);
		ret.append(prefix);
		ret.append(padding_width, '0');
		ret.append(code_string, actual_code_width);
		ret.append(suffix);
		return ret;
	}
};

static const mvs_error_category _mvs_error_category;

#ifdef __cpp_concepts
#define FUNCTION_TEMPLATE(FNAME, ARGSNAME, R) template< \
	typename FNAME, \
	typename... ARGSNAME\
> \
	requires std::is_invocable_r_v<R, FNAME, ARGSNAME...>
#else
#define FUNCTION_TEMPLATE(FNAME, ARGSNAME, R) template< \
	typename FNAME, \
	typename... ARGSNAME, \
	typename = std::enable_if_t<std::is_invocable_r_v<R, FNAME, ARGSNAME...>> \
>
#endif

FUNCTION_TEMPLATE(F, Args, int)
inline void _wrap_mvs(std::error_code& ec, F&& f, Args&&... args)
{
	if (auto ret = f(std::forward<Args>(args)...); ret != MV_OK)
		ec.assign(ret, _mvs_error_category);
}

FUNCTION_TEMPLATE(F, Args, int)
inline void _wrap_mvs(F&& f, Args&&... args)
{
	std::error_code ec;
	_wrap_mvs(ec, std::forward<F>(f), std::forward<Args>(args)...);
	if (ec)
		throw std::system_error(ec);
}

}

void mvs::_callback(unsigned char *data, ::MV_FRAME_OUT_INFO_EX *info, void *user)
{
	auto self = reinterpret_cast<mvs *>(user);
	const auto required_pixel_type = self->_colour ?
		::MvGvspPixelType::PixelType_Gvsp_BGR8_Packed :
		::MvGvspPixelType::PixelType_Gvsp_Mono8;

	cv::Mat image;
	if (info->enPixelType == required_pixel_type)
		cv::Mat(
			info->nHeight,
			info->nWidth,
			self->_colour ? CV_8UC3 : CV_8UC1,
			data
		).copyTo(image);
	else
	{
		image.create(info->nHeight, info->nWidth, self->_colour ? CV_8UC3 : CV_8UC1);

		::MV_CC_PIXEL_CONVERT_PARAM param;
		param.nWidth = info->nWidth;
		param.nHeight = info->nHeight;
		param.enSrcPixelType = info->enPixelType;
		param.pSrcData = data;
		param.nSrcDataLen = info->nFrameLen;
		param.enDstPixelType = required_pixel_type;
		param.pDstBuffer = image.data;
		param.nDstBufferSize = param.nDstLen = info->nHeight * info->nWidth;

		_wrap_mvs(::MV_CC_ConvertPixelType, self->_handle, &param);
	}

	auto guard = std::lock_guard(self->_lock);
	self->_images.push_back(std::move(image));
}

mvs::mvs(const ::MV_CC_DEVICE_INFO *device_info, bool colour) : _handle(nullptr), _colour(colour)
{
	_wrap_mvs(::MV_CC_CreateHandleWithoutLog, &_handle, device_info);
}

namespace
{

[[nodiscard]]
std::unordered_map<std::string_view, ::MV_CC_DEVICE_INFO *> _process_gig_e(::MV_CC_DEVICE_INFO_LIST& device_info_list)
{
	std::unordered_map<std::string_view, ::MV_CC_DEVICE_INFO *> ret;
	for (size_t i = 0; i < device_info_list.nDeviceNum; ++i)
	{
		auto device_info = device_info_list.pDeviceInfo[i];
		ret.emplace(reinterpret_cast<char *>(device_info->SpecialInfo.stGigEInfo.chSerialNumber), device_info);
	}
	return ret;
}

[[nodiscard]]
std::unordered_map<std::string_view, ::MV_CC_DEVICE_INFO *> _process_usb(::MV_CC_DEVICE_INFO_LIST& device_info_list)
{
	std::unordered_map<std::string_view, ::MV_CC_DEVICE_INFO *> ret;
	for (size_t i = 0; i < device_info_list.nDeviceNum; ++i)
	{
		auto device_info = device_info_list.pDeviceInfo[i];
		ret.emplace(reinterpret_cast<char *>(device_info->SpecialInfo.stUsb3VInfo.chSerialNumber), device_info);
	}
	return ret;
}

}

[[nodiscard]]
std::vector<std::unique_ptr<mvs>> mvs::find(
	const std::vector<std::string_view>& serials,
	transport_layer type,
	bool colour
)
{
	::MV_CC_DEVICE_INFO_LIST list;
	switch (type)
	{
		case transport_layer::USB:
			_wrap_mvs(::MV_CC_EnumDevices, MV_USB_DEVICE, &list);
			break;
		case transport_layer::GIG_E:
			_wrap_mvs(::MV_CC_EnumDevices, MV_GIGE_DEVICE, &list);
			break;
		default:
			return {};
	}

	std::vector<std::unique_ptr<mvs>> ret;
	if (serials.empty())
	{
		ret.reserve(list.nDeviceNum);
		for (size_t i = 0; i < list.nDeviceNum; ++i)
			ret.emplace_back(new mvs(list.pDeviceInfo[i], colour));
	}
	else
	{
		std::unordered_map<std::string_view, ::MV_CC_DEVICE_INFO *> mapping;
		switch (type)
		{
			case transport_layer::USB:
				mapping = _process_usb(list);
				break;
			case transport_layer::GIG_E:
				mapping = _process_gig_e(list);
				break;
			default:
				return {};
		}

		ret.reserve(serials.size());
		for (const auto& serial : serials)
			if (auto node = mapping.extract(serial))
				ret.emplace_back(new mvs(node.mapped(), colour));
	}
	return ret;
}

mvs::~mvs() noexcept
{
	_wrap_mvs(::MV_CC_DestroyHandle, _handle);
	_handle = nullptr;
}

void mvs::close()
{
	_wrap_mvs(::MV_CC_CloseDevice, _handle);
}

bool mvs::next_image(std::error_code& ec, cv::Mat& image)
{
	auto guard = std::lock_guard(_lock);
	if (_images.empty())
		return false;
	image = std::move(_images.front());
	_images.pop_front();
	return true;
}

void mvs::open()
{
	_wrap_mvs(::MV_CC_OpenDevice, _handle, MV_ACCESS_Control, 0);
}

std::string mvs::serial() const
{
	::MV_CC_DEVICE_INFO device_info;
	_wrap_mvs(::MV_CC_GetDeviceInfo, _handle, &device_info);
	switch (device_info.nTLayerType)
	{
		case MV_GIGE_DEVICE:
			return reinterpret_cast<char *>(device_info.SpecialInfo.stGigEInfo.chSerialNumber);
		case MV_USB_DEVICE:
			return reinterpret_cast<char *>(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber);
		default:
			return {};
	}
}

void mvs::start(bool latest_only)
{
	_wrap_mvs(
		::MV_CC_SetGrabStrategy,
		_handle,
		latest_only ?
			::MV_GRAB_STRATEGY::MV_GrabStrategy_LatestImagesOnly :
			::MV_GRAB_STRATEGY::MV_GrabStrategy_OneByOne
	);
	_wrap_mvs(::MV_CC_StartGrabbing, _handle);
}

void mvs::stop()
{
	_wrap_mvs(::MV_CC_StopGrabbing, _handle);
}

void mvs::subscribe()
{
	_wrap_mvs(::MV_CC_RegisterImageCallBackEx, _handle, _callback, this);
}

void mvs::unsubscribe()
{
	_wrap_mvs(::MV_CC_RegisterImageCallBackEx, _handle, nullptr, nullptr);
}

namespace
{

#ifdef __cpp_concepts
#define PARAMETER_CONSTRAINT(N, C) \
template<typename N> \
	requires C<N>
#else
#define PARAMETER_CONSTRAINT(N, C) \
template< \
	typename N, \
	const std::enable_if_t<C<N>, N> * = nullptr \
>
#endif

PARAMETER_CONSTRAINT(T, std::is_integral_v)
inline bool _mvs_set(void *handle, const char* name, T value)
{
	std::error_code ec;
	_wrap_mvs(ec, ::MV_CC_SetIntValueEx, handle, name, value);
	return !ec;
}

PARAMETER_CONSTRAINT(T, std::is_floating_point_v)
inline bool _mvs_set(void *handle, const char* name, T value)
{
	std::error_code ec;
	_wrap_mvs(ec, ::MV_CC_SetFloatValue, handle, name, value);
	return !ec;
}

template<bool string>
inline bool _mvs_set(void *handle, const char *name, const char *value)
{
	std::error_code ec;
	if constexpr (string)
		_wrap_mvs(ec, ::MV_CC_SetStringValue, handle, name, value);
	else
		_wrap_mvs(ec, ::MV_CC_SetEnumValueByString, handle, name, value);
	return !ec;
}

}

[[nodiscard]]
bool mvs::set_exposure_time(const std::chrono::duration<double, std::micro>& time)
{
	return _mvs_set<false>(_handle, "ExposureAuto", "Off") &&
		_mvs_set(_handle, "ExposureTime", time.count()) &&
		_mvs_set<false>(_handle, "ExposureMode", "Timed");
}

[[nodiscard]]
bool mvs::set_gain(double gain)
{
	return _mvs_set<false>(_handle, "GainAuto", "Off") &&
		_mvs_set(_handle, "Gain", gain);
}

namespace
{

static constexpr auto _universal_lines = std::array {
	"Line0",
	"Line1",
	"Line2"
};

}

[[nodiscard]]
bool mvs::set_line_debouncer_time(size_t line, const std::chrono::microseconds& time)
{
	return line < _universal_lines.size() &&
		_mvs_set<false>(_handle, "LineSelector", _universal_lines[line]) &&
		_mvs_set(_handle, "LineDebouncerTime", time.count());
}

namespace
{

static constexpr auto _trigger_lines = std::array {
	"Line0",
	"Line1",
	"Line2"
};

}

[[nodiscard]]
bool mvs::set_manual_trigger_line_source(size_t line, const std::chrono::duration<double, std::micro>& delay)
{
	return line < _trigger_lines.size() &&
		_mvs_set<false>(_handle, "TriggerSource", "Line0") &&
		_mvs_set<false>(_handle, "TriggerActivation", "RisingEdge") &&
		_mvs_set(_handle, "TriggerDelay", delay.count()) &&
		_mvs_set<false>(_handle, "TriggerMode", "On");
}

}
