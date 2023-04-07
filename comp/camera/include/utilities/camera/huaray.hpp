#ifndef __UTILITIES_CAMERA_HUARAY_HPP__
#define __UTILITIES_CAMERA_HUARAY_HPP__

#include <cstddef>

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <IMVApi.h>
#include <opencv2/core.hpp>

#include "utilities/camera/base.hpp"

namespace utilities::camera
{

struct huaray final : public base::device
{
	enum class transport_layer
	{
		USB,
		GIG_E
	};
private:
	static void _callback(::IMV_Frame *frame, void *user);

	IMV_HANDLE _handle;
	bool _colour;
	base::rotation_direction _rotation;
	std::mutex _lock;
	std::list<cv::Mat> _images;
	size_t _counter;

	huaray(unsigned int index, bool colour);
public:
	[[nodiscard]]
	static std::vector<std::unique_ptr<huaray>> find(
		const std::vector<std::string>& serials,
		transport_layer type,
		bool colour
	);

	virtual ~huaray() noexcept override;

	// base::device

	[[nodiscard]]
	inline virtual base::brand brand() const override
	{
		return base::brand::HUARAY;
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
};

}

#endif
