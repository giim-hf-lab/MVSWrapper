get_ext_version(paddle __L_VERSION)

include("vars/${__L_VERSION}/detections.cmake" NO_POLICY_SCOPE)

string(JOIN "/" __L_URL
	"https://paddle-inference-lib.bj.bcebos.com"
	"${__L_VERSION}"
	"cxx_c"
	"${__L_PLATFORM}"
	"${__L_BUILD}"
	"paddle_inference.zip"
)

FetchContent_Declare(paddle_inference
	URL "${__L_URL}"
	URL_MD5 "${__L_MD5}"
	DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Populate(paddle_inference)

FetchContent_GetProperties(paddle_inference SOURCE_DIR __L_SOURCE_DIR)
set(__L_CORE_DIR "${__L_SOURCE_DIR}/paddle")
set(__L_CORE_INCLUDE_DIR "${__L_CORE_DIR}/include")
set(__L_CORE_LIBRARY_DIR "${__L_CORE_DIR}/lib")

add_library(paddle::inference SHARED IMPORTED GLOBAL)
set_property(
	TARGET
		paddle::inference
	PROPERTY IMPORTED_LOCATION "${__L_CORE_LIBRARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}paddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
if (WIN32)
	set_property(
		TARGET
			paddle::inference
		PROPERTY IMPORTED_IMPLIB "${__L_CORE_LIBRARY_DIR}/paddle_inference.lib"
	)
	file(GLOB_RECURSE __L_ALL_DLLS
		LIST_DIRECTORIES FALSE
		RELATIVE "${__L_SOURCE_DIR}"
		"${__L_SOURCE_DIR}/*.dll"
	)
	set(__L_ALL_OUTPUT_DLLS)
	foreach (__L_DLL ${__L_ALL_DLLS})
		cmake_path(GET __L_DLL FILENAME __L_DLL_FILENAME)
		list(APPEND __L_ALL_OUTPUT_DLLS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${__L_DLL_FILENAME}")
	endforeach ()
	add_custom_target(paddle-inference-dlls
			"${CMAKE_COMMAND}"
			"-E"
			"copy_if_different"
			${__L_ALL_DLLS}
			"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
		BYPRODUCTS
			${__L_ALL_OUTPUT_DLLS}
		WORKING_DIRECTORY "${__L_SOURCE_DIR}"
		VERBATIM
		USES_TERMINAL
		COMMAND_EXPAND_LISTS
	)
	add_dependencies(paddle::inference paddle-inference-dlls)
endif ()
set_property(
	TARGET
		paddle::inference
	PROPERTY INTERFACE_INCLUDE_DIRECTORIES
		"${__L_CORE_INCLUDE_DIR}"
)

set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "paddle::inference")
