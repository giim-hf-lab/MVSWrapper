set(onnxruntime_USE_CUDA ON)
set(onnxruntime_BUILD_UNIT_TESTS OFF)
set(onnxruntime_USE_AVX ON)
set(onnxruntime_USE_AVX2 ON)
set(onnxruntime_BUILD_SHARED_LIB ON)
set(onnxruntime_ENABLE_LTO ON)
set(onnxruntime_USE_FULL_PROTOBUF OFF)
set(onnxruntime_DISABLE_RTTI ON)

add_subdirectory("${SUBMODULE}/cmake" EXCLUDE_FROM_ALL)

if (MSVC)
	# onnxruntime/core/framework/bfc_arena.cc triggers an MSVC internal compiler error in lines range from 107 to 127
	# due to complex lambda expressions. Adding `/Zc:lambda-` forces the compiler to use the old lambda processor
	# which works correctly.
	target_compile_options(onnxruntime_framework
		PRIVATE
			"/Zc:lambda-"
	)
endif ()
target_include_directories(onnxruntime SYSTEM
	INTERFACE
		"${CMAKE_CURRENT_SOURCE_DIR}/${SUBMODULE}/include"
)
