#include <cstddef>

#include <filesystem>
#include <utility>
#include <vector>

#include "filecache.hpp"

namespace utilities
{

filecache::filecache(std::filesystem::path path, size_t expansion_lines) :
	_path(std::move(path)),
	_mmap(_path.native()),
	_block_sizes(),
	_expansion_lines(expansion_lines)
{}

filecache::filecache(std::filesystem::path path, std::vector<size_t> block_sizes, size_t expansion_lines) :
	_path(std::move(path)),
	_mmap(_path.native()),
	_block_sizes(std::move(block_sizes)),
	_expansion_lines(expansion_lines)
{}

}
