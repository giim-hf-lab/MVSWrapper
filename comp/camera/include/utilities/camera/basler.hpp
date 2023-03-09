#ifndef __UTILITIES_CAMERA_BASLER_HPP__
#define __UTILITIES_CAMERA_BASLER_HPP__

#include <cstddef>

#include <chrono>
#include <list>
#include <memory>
#include <mutex>
#include <string>
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
		ANY,
		USB,
		GIG_E
	};
private:
	struct initialiser final
	{
		initialiser();

		~initialiser() noexcept;
	};

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

	static const initialiser _global_guard;

	Pylon::CBaslerUniversalInstantCamera _instance;
	image_listener _listener;

	basler(const Pylon::CDeviceInfo& device_info, bool colour);
public:
	[[nodiscard]]
	static std::vector<std::unique_ptr<basler>> find(
		const std::vector<std::string_view>& serials,
		transport_layer type,
		bool colour
	);

	// base::device

	virtual void close() override;

	[[nodiscard]]
	virtual bool next_image(std::error_code& ec, cv::Mat& image) override;

	virtual void open() override;

	virtual std::string serial() const override;

	virtual void start(bool latest_only) override;

	virtual void stop() override;

	virtual void subscribe() override;

	virtual void unsubscribe() override;

	// exposure

	[[nodiscard]]
	bool set_exposure_time(const std::chrono::duration<double, std::micro>& time);

	template<typename Rep, typename Period>
	[[nodiscard]]
	inline bool set_exposure_time(const std::chrono::duration<Rep, Period>& time)
	{
		return set_exposure_time(std::chrono::duration<double, std::micro> { time });
	}

	// gain

	[[nodiscard]]
	bool set_gain(double gain);

	// line debouncer time

	[[nodiscard]]
	bool set_line_debouncer_time(size_t line, const std::chrono::duration<double, std::micro>& time);

	template<typename Rep, typename Period>
	[[nodiscard]]
	inline bool set_line_debouncer_time(size_t line, const std::chrono::duration<Rep, Period>& time)
	{
		return set_line_debouncer_time(line, std::chrono::duration<double, std::micro> { time });
	}

	// manual trigger

	[[nodiscard]]
	bool set_manual_trigger_line_source(size_t line, const std::chrono::duration<double, std::micro>& delay);

	template<typename Rep, typename Period>
	[[nodiscard]]
	inline bool set_manual_trigger_line_source(size_t line, const std::chrono::duration<Rep, Period>& delay)
	{
		return set_manual_trigger_line_source(line, std::chrono::duration<double, std::micro> { delay });
	}
};

}

#endif
