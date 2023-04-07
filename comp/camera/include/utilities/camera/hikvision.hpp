#ifndef __UTILITIES_CAMERA_HIKVISION_HPP__
#define __UTILITIES_CAMERA_HIKVISION_HPP__

#include <cstddef>

#include <chrono>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <MvCameraControl.h>
#include <opencv2/core.hpp>

#include "utilities/camera/base.hpp"

namespace utilities::camera
{

struct hikvision final : public base::device
{
	enum class transport_layer
	{
		USB,
		GIG_E
	};
private:
	static void _callback(unsigned char *data, ::MV_FRAME_OUT_INFO_EX *info, void *user);

	void *_handle;
	bool _colour;
	base::rotation_direction _rotation;
	std::mutex _lock;
	std::list<cv::Mat> _images;
	size_t _counter;

	hikvision(const ::MV_CC_DEVICE_INFO *device_info, bool colour);
public:
	[[nodiscard]]
	static std::vector<std::unique_ptr<hikvision>> find(
		const std::vector<std::string>& serials,
		transport_layer type,
		bool colour
	);

	virtual ~hikvision() noexcept override;

	// base::device

	[[nodiscard]]
	inline virtual base::brand brand() const override
	{
		return base::brand::HIKVISION;
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
		return set_exposure_time(std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(time));
	}

	// gain

	[[nodiscard]]
	bool set_gain(double gain);

	// line debouncer time

	[[nodiscard]]
	bool set_line_debouncer_time(size_t line, const std::chrono::microseconds& time);

	template<typename Rep, typename Period>
	[[nodiscard]]
	inline bool set_line_debouncer_time(size_t line, const std::chrono::duration<Rep, Period>& time)
	{
		return set_line_debouncer_time(line, std::chrono::duration_cast<std::chrono::microseconds>(time));
	}

	// line output

	[[nodiscard]]
	bool output_signal(size_t line, const std::chrono::microseconds& duration);

	// manual trigger

	[[nodiscard]]
	bool set_manual_trigger_line_source(size_t line, const std::chrono::duration<double, std::micro>& delay);

	template<typename Rep1, typename Period1, typename Rep2, typename Period2>
	[[nodiscard]]
	inline bool set_manual_trigger_line_source(size_t line, const std::chrono::duration<Rep1, Period1>& delay)
	{
		return set_manual_trigger_line_source(
			line,
			std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(delay)
		);
	}
};

}

#endif
