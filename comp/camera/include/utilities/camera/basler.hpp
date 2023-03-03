#ifndef __UTILITIES_CAMERA_BASLER_HPP__
#define __UTILITIES_CAMERA_BASLER_HPP__

#include <list>
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

struct basler final : public base::reader
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
public:
	basler(const std::string_view& serial, transport_layer type, bool colour);

	void close();

	void open();

	void start(bool latest_only = false);

	void stop();

	void subscribe(bool exclusive = true);

	bool unsubscribe();

	[[nodiscard]]
	virtual bool next_image(std::error_code& ec, cv::Mat& image) override;
};

}

#endif
