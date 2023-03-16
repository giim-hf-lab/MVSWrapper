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
		inline initialiser()
		{
			Pylon::PylonInitialize();
		}

		inline ~initialiser() noexcept
		{
			Pylon::PylonTerminate();
		}
	};

	class image_listener final : public Pylon::CBaslerUniversalImageEventHandler
	{
		Pylon::CImageFormatConverter _converter;
		bool _colour;
		base::rotation_direction _rotation;
		std::mutex _lock;
		std::list<cv::Mat> _images;
		size_t _counter;
	public:
		image_listener(bool colour);

		[[nodiscard]]
		inline base::frame next(std::error_code& ec);

		[[nodiscard]]
		inline base::rotation_direction rotation() const;

		inline void rotation(base::rotation_direction direction);

		virtual void OnImageEventHandlerRegistered(Pylon::CBaslerUniversalInstantCamera& camera) override;

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

	[[nodiscard]]
	inline virtual base::brand brand() const override
	{
		return base::brand::BASLER;
	}

	virtual void close() override;

	[[nodiscard]]
	virtual base::frame next_image(std::error_code& ec) override;

	virtual void open() override;

	[[nodiscard]]
	virtual base::rotation_direction rotation() const override;

	virtual void rotation(base::rotation_direction direction) override;

	[[nodiscard]]
	virtual std::string serial() const override;

	virtual void start() override;

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

	// line output

	[[nodiscard]]
	bool output_signal(size_t line);

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
