#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <concepts>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <clipper2/clipper.core.h>
#include <clipper2/clipper.offset.h>
#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <mio/mmap.hpp>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
