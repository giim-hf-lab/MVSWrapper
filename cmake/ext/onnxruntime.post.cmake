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
		"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${__F_SUBMODULE}/include>"
)
