prefixed_option(EXPORT_PADDLE OFF "Export PaddlePaddle targets.")

get_prefixed_option(EXPORT_PADDLE OUTPUT_PREFIXES "__G")

if (__G_EXPORT_PADDLE)
	prefixed_option(PADDLE_INFEFERNCE_VERSION "" "The version of PaddlePaddle inference engine to be downloaded."
		REFERENCE
		TYPE STRING
	)
	prefixed_option(PADDLE_INFEFERNCE_BUILD "" "The PaddlePaddle inference engine build to be downloaded."
		REFERENCE
		TYPE STRING
	)
	prefixed_option(PADDLE_INFEFERNCE_BUILD_MD5 "" "The PaddlePaddle inference engine build hash to skip downloading."
		REFERENCE
		TYPE STRING
	)

	if (WIN32)
		set(__L_PLATFORM "Windows")
	elseif (UNIX AND APPLE)
		set(__L_PLATFORM "MacOS")
	elseif (UNIX)
		set(__L_PLATFORM "Linux")
	else ()
		project_message(STATUS "Unknown platform ${CMAKE_HOST_SYSTEM_NAME} for PaddlePaddle inference.")
	endif ()

	get_prefixed_option(PADDLE_INFEFERNCE_VERSION OUTPUT_PREFIXES "__L")
	get_prefixed_option(PADDLE_INFEFERNCE_BUILD OUTPUT_PREFIXES "__L")
	get_prefixed_option(PADDLE_INFEFERNCE_BUILD_MD5 OUTPUT_PREFIXES "__L")

	string(JOIN "/" __L_PADDLE_INFERENCE_URL
		"https://paddle-inference-lib.bj.bcebos.com"
		"${__L_PADDLE_INFEFERNCE_VERSION}"
		"cxx_c"
		"${__L_PLATFORM}"
		"${__L_PADDLE_INFEFERNCE_BUILD}"
		"paddle_inference.zip"
	)

	FetchContent_Declare(paddle_inference
		URL "${__L_PADDLE_INFERENCE_URL}"
		URL_MD5 "${__L_PADDLE_INFEFERNCE_BUILD_MD5}"
		DOWNLOAD_EXTRACT_TIMESTAMP TRUE
	)
	FetchContent_Populate(paddle_inference)

	set(PADDLE_INFEFERNCE_VERSION "${__L_PADDLE_INFEFERNCE_VERSION}" PARENT_SCOPE)

	FetchContent_GetProperties(paddle_inference SOURCE_DIR __L_PADDLE_INFERENCE_DIR)
	set(__L_PADDLE_INFERENCE_CORE_DIR "${__L_PADDLE_INFERENCE_DIR}/paddle")
	set(__L_PADDLE_INFERENCE_CORE_INCLUDE_DIR "${__L_PADDLE_INFERENCE_CORE_DIR}/include")
	set(__L_PADDLE_INFERENCE_CORE_LIBRARY_DIR "${__L_PADDLE_INFERENCE_CORE_DIR}/lib")

	add_library(pre::paddle::shared SHARED IMPORTED GLOBAL)
	set_property(
		TARGET
			pre::paddle::shared
		PROPERTY SYSTEM TRUE
	)
	if (WIN32)
		set_property(
			TARGET
				pre::paddle::shared
			PROPERTY IMPORTED_LOCATION "${__L_PADDLE_INFERENCE_CORE_LIBRARY_DIR}/paddle_inference.dll"
		)
		set_property(
			TARGET
				pre::paddle::shared
			PROPERTY IMPORTED_IMPLIB "${__L_PADDLE_INFERENCE_CORE_LIBRARY_DIR}/paddle_inference.lib"
		)
		file(GLOB_RECURSE __L_ALL_DLLS
			LIST_DIRECTORIES FALSE
			RELATIVE "${__L_PADDLE_INFERENCE_DIR}"
			"${__L_PADDLE_INFERENCE_DIR}/*.dll"
		)
		set(__L_ALL_OUTPUT_DLLS)
		foreach (__L_DLL ${__L_ALL_DLLS})
			cmake_path(GET __L_DLL FILENAME __L_DLL_FILENAME)
			list(APPEND __L_ALL_OUTPUT_DLLS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${__L_DLL_FILENAME}")
		endforeach ()
		add_custom_target(pre-paddle-shared-_dlls
				"${CMAKE_COMMAND}"
				"-E"
				"copy_if_different"
				${__L_ALL_DLLS}
				"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
			BYPRODUCTS
				${__L_ALL_OUTPUT_DLLS}
			WORKING_DIRECTORY "${__L_PADDLE_INFERENCE_DIR}"
			VERBATIM
			USES_TERMINAL
			COMMAND_EXPAND_LISTS
		)
		add_dependencies(pre::paddle::shared pre-paddle-shared-_dlls)
	elseif (UNIX AND APPLE)
		set_property(
			TARGET
				pre::paddle::shared
			PROPERTY IMPORTED_LOCATION "${__L_PADDLE_INFERENCE_CORE_LIBRARY_DIR}/libpaddle_inference.dylib"
		)
	elseif (UNIX)
		set_property(
			TARGET
				pre::paddle::shared
			PROPERTY IMPORTED_LOCATION "${__L_PADDLE_INFERENCE_CORE_LIBRARY_DIR}/libpaddle_inference.so"
		)
	else ()
		project_message(STATUS "Unknown platform ${CMAKE_HOST_SYSTEM_NAME} for PaddlePaddle inference.")
	endif ()
	set_property(
		TARGET
			pre::paddle::shared
		PROPERTY INTERFACE_INCLUDE_DIRECTORIES
			"${__L_PADDLE_INFERENCE_CORE_INCLUDE_DIR}"
	)
endif ()
