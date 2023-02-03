find_package(Python 3.8 REQUIRED COMPONENTS Interpreter GLOBAL)

if (WIN32)
	if (${OPTIONS_PREFIX}_SUPERDOG_BASE_DIR)
		cmake_path(IS_ABSOLUTE ${OPTIONS_PREFIX}_SUPERDOG_BASE_DIR SUPERDOG_BASE_DIR_IS_ABSOLUTE)
		if (NOT SUPERDOG_BASE_DIR_IS_ABSOLUTE)
			project_message(FATAL_ERROR "Incorrect input path of the base directory containing the SuperDog libraries (${${OPTIONS_PREFIX}_SUPERDOG_BASE_DIR} is not absolute).")
		endif ()
		set(SUPERDOG_BASE_DIR "${${OPTIONS_PREFIX}_SUPERDOG_BASE_DIR}")
	else ()
		if (DEFINED ENV{PROGRAMFILES\(x86\)})
			set(PROGRAMFILES_BASE "$ENV{PROGRAMFILES\(x86\)}")
		elseif (DEFINED ENV{PROGRAMFILES})
			set(PROGRAMFILES_BASE "$ENV{PROGRAMFILES}")
		else ()
			message(FATAL_ERROR "Unknown environment detected.")
		endif ()
		cmake_path(
			APPEND PROGRAMFILES_BASE
				"Gemalto"
				"SuperDog"
			OUTPUT_VARIABLE SUPERDOG_BASE_DIR
		)
	endif ()

	if (NOT IS_DIRECTORY "${SUPERDOG_BASE_DIR}")
		message(FATAL_ERROR "SuperDog cannot be found (${SUPERDOG_BASE_DIR} is not a directory).")
	endif ()

	if (LEGACY_ARCHITECTURE)
		set(SUPERDOG_ARCHITECTURE "win32")
		set(SUPERDOG_LIBRARY_SUFFIX)
	else ()
		set(SUPERDOG_ARCHITECTURE "x64")
		set(SUPERDOG_LIBRARY_SUFFIX "_x64")
	endif ()
else ()
	message(FATAL_ERROR "SuperDog wrapper not supported on ${CMAKE_HOST_SYSTEM_NAME}.")
endif ()

if (SUPERDOG_VERSION)
	cmake_path(APPEND SUPERDOG_BASE_DIR "${SUPERDOG_VERSION}")
	if (NOT IS_DIRECTORY "${SUPERDOG_BASE_DIR}")
		message(FATAL_ERROR "SuperDog version ${SUPERDOG_VERSION} cannot be found (${SUPERDOG_BASE_DIR} is not a directory).")
	endif ()
else ()
	file(GLOB SUPERDOG_VERSION_CANDIDATES RELATIVE "${SUPERDOG_BASE_DIR}" "${SUPERDOG_BASE_DIR}/*")
	if (NOT SUPERDOG_VERSION_CANDIDATES)
		message(FATAL_ERROR "SuperDog default version selection failed (${SUPERDOG_BASE_DIR} is empty).")
	endif ()
	set(SUPERDOG_VERSION)
	foreach (CANDIDATE IN LISTS SUPERDOG_VERSION_CANDIDATES)
		if (IS_DIRECTORY "${SUPERDOG_BASE_DIR}/${CANDIDATE}" AND CANDIDATE MATCHES "^[0-9]+\\.[0-9]+$" AND CANDIDATE VERSION_GREATER SUPERDOG_VERSION)
			set(SUPERDOG_VERSION "${CANDIDATE}")
		endif ()
	endforeach ()
	if (NOT SUPERDOG_VERSION)
		message(FATAL_ERROR "SuperDog default version selection failed (unknown version pattern).")
	endif ()
	cmake_path(APPEND SUPERDOG_BASE_DIR "${SUPERDOG_VERSION}")
endif ()

project_message(STATUS "Using SuperDog version ${SUPERDOG_VERSION}.")

cmake_path(
	APPEND SUPERDOG_BASE_DIR
		"API"
		"Licensing"
		"C"
		"${SUPERDOG_ARCHITECTURE}"
)
if (NOT IS_DIRECTORY "${SUPERDOG_BASE_DIR}")
	message(FATAL_ERROR "SuperDog API directory cannot be found (${SUPERDOG_BASE_DIR} is not a directory).")
endif ()

# 	set(USER_HOME "$ENV{USERPROFILE}")
# 	cmake_path(
# 		APPEND USER_HOME
# 			"Documents"
# 			"Gemalto"
# 			"SuperDog ${GEMALTO_SUPERDOG_VERSION}"
# 			"API"
# 			"Licensing"
# 			"C"
# 			"${GEMALTO_ARCHITECTURE}"
# 		OUTPUT_VARIABLE GEMALTO_CUSTOMISED_SUPERDOG_LICENSING_API_DIR
# 	)
# 	find_file(GEMALTO_SUPERDOG_LICENSING_API_LIBRARY
# 		"libdog_windows${GEMALTO_LIBRARY_SUFFIX}_${GEMALTO_SUPERDOG_BUILD}.lib"
# 		HINTS
# 			"${GEMALTO_CUSTOMISED_SUPERDOG_LICENSING_API_DIR}"
# 			"${GEMALTO_SUPERDOG_LICENSING_API_DIR}"
# 		NO_CACHE
# 		REQUIRED
# 		NO_DEFAULT_PATH
# 	)

if (NOT SUPERDOG_BUILD)
	project_message(STATUS "Gemalto SuperDog build defaults to \"demo\".")
	set(S_UPERDOG_BUILD "demo")
endif ()

if (NOT SUPERDOG_VENDOR_CODE_LABEL)
	project_message(STATUS "Gemalto SuperDog vendor code label defaults to \"DEMOMA\".")
	set(SUPERDOG_VENDOR_CODE_LABEL "DEMOMA")
endif ()

# add_library(gemalto_superdog_api STATIC IMPORTED)
# set_property(
# 	TARGET
# 		gemalto_superdog_api
# 	PROPERTY IMPORTED_LOCATION "${GEMALTO_SUPERDOG_LICENSING_API_LIBRARY}"
# )
# set_property(
# 	TARGET
# 		gemalto_superdog_api
# 	PROPERTY SYSTEM TRUE
# )
# target_include_directories(gemalto_superdog_api INTERFACE
# 	"${GEMALTO_CUSTOMISED_SUPERDOG_LICENSING_API_DIR}"
# 	"${GEMALTO_SUPERDOG_LICENSING_API_DIR}"
# )
# target_link_libraries(gemalto_superdog_api INTERFACE
# 	legacy_stdio_definitions.lib
# )

# cmake_path(
# 	APPEND CMAKE_CURRENT_SOURCE_DIR
# 		"vendor_code.py"
# 	OUTPUT_VARIABLE GEMALTO_VENDOR_CODE_GENERATOR
# )
# cmake_path(
# 	APPEND GEMALTO_SUPERDOG_BASE
# 		"VendorCodes"
# 		"${GEMALTO_SUPERDOG_VENDOR_CODE_LABEL}.hvc"
# 	OUTPUT_VARIABLE GEMALTO_SUPERDOR_VENDOR_CODE_FILE
# )
# cmake_path(
# 	APPEND GENERATED_FILES_DIR
# 		"gemalto"
# 		"vendor_codes"
# 	OUTPUT_VARIABLE GEMALTO_VENDOR_CODE_CPP_DIR
# )
# cmake_path(
# 	APPEND GEMALTO_VENDOR_CODE_CPP_DIR
# 		"superdog.vc"
# 	OUTPUT_VARIABLE GEMALTO_SUPERDOR_VENDOR_CODE_CPP
# )
# file(MAKE_DIRECTORY "${GEMALTO_VENDOR_CODE_CPP_DIR}")
# list(APPEND GEMALTO_SUPERDOG_VENDOR_CODE_GENERATOR
# 	"${Python_EXECUTABLE}"
# 	"${GEMALTO_VENDOR_CODE_GENERATOR}"
# 	"--name" "SUPERDOG"
# 	"--input-file" "${GEMALTO_SUPERDOR_VENDOR_CODE_FILE}"
# 	"--output-file" "${GEMALTO_SUPERDOR_VENDOR_CODE_CPP}"
# )
# execute_process(
# 	COMMAND
# 		${GEMALTO_SUPERDOG_VENDOR_CODE_GENERATOR}
# 	WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
# 	ENCODING UTF-8
# 	COMMAND_ERROR_IS_FATAL ANY
# )
# add_custom_command(
# 	OUTPUT
# 		"${GEMALTO_SUPERDOR_VENDOR_CODE_CPP}"
# 	COMMAND
# 		${GEMALTO_SUPERDOG_VENDOR_CODE_GENERATOR}
# 	DEPENDS
# 		"${GEMALTO_SUPERDOR_VENDOR_CODE_FILE}"
# 		"${GEMALTO_VENDOR_CODE_GENERATOR}"
# 	WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
# 	COMMENT "Generating Gemalto vendor code for SuperDog"
# 	VERBATIM
# 	USES_TERMINAL
# )
# add_custom_target(gemalto_superdog_vendor_code
# 	DEPENDS
# 		"${GEMALTO_SUPERDOR_VENDOR_CODE_CPP}"
# )

# if (MSVC)
# 	create_msvc_cxx_module_target(gemalto
# 		INCLUDE_DIRECTORIES
# 			"${CMAKE_CURRENT_SOURCE_DIR}/include"
# 		LINK_LIBRARIES
# 			gemalto_superdog_api
# 	)
# 	add_dependencies(gemalto gemalto_superdog_vendor_code)
# 	add_library(gemalto::gemalto ALIAS gemalto)
# else ()
# 	message(FATAL_ERROR "Gemalto wrapper requires C++20 module support")
# endif ()
# 
# add_executable(gemalto_driver EXCLUDE_FROM_ALL main.cpp)
# target_link_libraries(gemalto_driver
# 	PRIVATE
# 		gemalto::gemalto
# )
