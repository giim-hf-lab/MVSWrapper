#ifndef __UTILITIES_CAMERA_BASLER_HPP__
#define __UTILITIES_CAMERA_BASLER_HPP__

#include <list>
#include <memory>
#include <mutex>
#include <string_view>
#include <system_error>
#include <vector>

#include <opencv2/core.hpp>
#include <pylon/BaslerUniversalInstantCamera.h>
#include <pylon/PylonIncludes.h>

#include "utilities/camera/base.hpp"

namespace utilities::camera
{

struct basler final : public base::device
{
	enum class transport_layer
	{
		USB,
		GIG_E
	};

	struct initialiser final
	{
		initialiser();

		~initialiser() noexcept;
	};
private:
	class image_listener final : public Pylon::CBaslerUniversalImageEventHandler
	{
		Pylon::CImageFormatConverter _converter;
		bool _colour;
		std::list<cv::Mat> _images;
		std::mutex _lock;
	public:
		image_listener(bool colour);

		~image_listener() noexcept;

		inline bool next(std::error_code& ec, cv::Mat& image);

		virtual void OnImageGrabbed(
			Pylon::CBaslerUniversalInstantCamera& camera,
			const Pylon::CBaslerUniversalGrabResultPtr& grabResult
		) override;
	};

	Pylon::CBaslerUniversalInstantCamera _instance;
	image_listener _listener;

	basler(const Pylon::CDeviceInfo& device_info, bool colour);
public:
	[[nodiscard]]
	static std::vector<std::unique_ptr<base::device>> find(
		const std::vector<std::string_view>& serials,
		transport_layer type,
		bool colour
	);

	virtual void close() override;

	virtual void open() override;

	virtual void start(bool latest_only) override;

	virtual void stop() override;

	virtual void subscribe(bool exclusive) override;

	virtual void unsubscribe() override;

	[[nodiscard]]
	virtual bool next_image(std::error_code& ec, cv::Mat& image) override;
};

}

#endif
