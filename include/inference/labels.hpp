#ifndef __INFERENCE_UTILS_HPP__
#define __INFERENCE_UTILS_HPP__

#include <cstddef>

#include <algorithm>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <mio/mmap.hpp>

namespace inference
{

class labels_mapper final
{
	std::vector<std::string_view> _labels;
	std::unordered_map<std::string, size_t> _indices;

	labels_mapper() = default;
public:
	static labels_mapper from_file(const std::string & labels_path)
	{
		labels_mapper ret;
		ret._indices.max_load_factor(1);

		auto labels_mmap = mio::mmap_source(labels_path);
		std::string buffer(labels_mmap.data(), labels_mmap.size());
		for (size_t pos = 0, new_pos = 0, i = 0; new_pos < buffer.size() && pos < buffer.size(); pos = new_pos + 1)
		{
			new_pos = std::min(buffer.size(), buffer.find('\n', pos));
			std::string s(buffer.c_str() + pos, new_pos - pos);
			if (auto [it, emplaced] = ret._indices.try_emplace(std::move(s), i); emplaced)
			{
				ret._labels.emplace_back(it->first);
				++i;
			}
		}
		ret._labels.shrink_to_fit();

		return ret;
	}

	~labels_mapper() = default;

	labels_mapper(const labels_mapper &) = default;
	labels_mapper(labels_mapper &&) noexcept = default;

	labels_mapper & operator=(const labels_mapper &) & = default;
	labels_mapper & operator=(labels_mapper &&) & noexcept = default;

	labels_mapper & operator=(const labels_mapper &) && = delete;
	labels_mapper & operator=(labels_mapper &&) && noexcept = delete;

	[[nodiscard]]
	const auto & at(const std::string & label) const &
	{
		return _indices.at(label);
	}

	[[nodiscard]]
	const auto & at(size_t index) const &
	{
		return _labels.at(index);
	}

	[[nodiscard]]
	const auto & indices() const & noexcept
	{
		return _indices;
	}

	[[nodiscard]]
	const auto & labels() const & noexcept
	{
		return _labels;
	}

	[[nodiscard]]
	size_t size() const & noexcept
	{
		return _labels.size();
	}
};

}

#endif
