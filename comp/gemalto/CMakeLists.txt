if (NOT GEMALTO_SUPERDOG_VERSION)
	set(GEMALTO_SUPERDOG_VERSION "2.5")
endif ()

if (NOT GEMALTO_SUPERDOG_BUILD)
	message(STATUS "Gemalto SuperDog build defaults to \"demo\".")
	set(GEMALTO_SUPERDOG_BUILD "demo")
endif ()

if (NOT GEMALTO_SUPERDOG_VENDOR_CODE_LABEL)
	message(STATUS "Gemalto SuperDog vendor code label defaults to \"DEMOMA\".")
	set(GEMALTO_SUPERDOG_VENDOR_CODE_LABEL "DEMOMA")
endif ()

if (WIN32)
	if (DEFINED ENV{PROGRAMFILES\(x86\)})
		set(PROGRAMFILES_BASE "$ENV{PROGRAMFILES\(x86\)}")
		set(GEMALTO_ARCHITECTURE "x64")
		set(GEMALTO_LIBRARY_SUFFIX "_x64")
	elseif (DEFINED ENV{PROGRAMFILES})
		set(PROGRAMFILES_BASE "$ENV{PROGRAMFILES}")
		set(GEMALTO_ARCHITECTURE "win32")
	else ()
		message(FATAL_ERROR "Unknown environment detected.")
	endif ()

	cmake_path(
		APPEND PROGRAMFILES_BASE
			"Gemalto"
			"SuperDog"
			"${GEMALTO_SUPERDOG_VERSION}"
		OUTPUT_VARIABLE GEMALTO_SUPERDOG_BASE
	)
	cmake_path(
		APPEND GEMALTO_SUPERDOG_BASE
			"API"
			"Licensing"
			"C"
			"${GEMALTO_ARCHITECTURE}"
		OUTPUT_VARIABLE GEMALTO_SUPERDOG_LICENSING_API_DIR
	)
	set(USER_HOME "$ENV{USERPROFILE}")
	cmake_path(
		APPEND USER_HOME
			"Documents"
			"Gemalto"
			"SuperDog ${GEMALTO_SUPERDOG_VERSION}"
			"API"
			"Licensing"
			"C"
			"${GEMALTO_ARCHITECTURE}"
		OUTPUT_VARIABLE GEMALTO_CUSTOMISED_SUPERDOG_LICENSING_API_DIR
	)
	find_file(GEMALTO_SUPERDOG_LICENSING_API_LIBRARY
		"libdog_windows${GEMALTO_LIBRARY_SUFFIX}_${GEMALTO_SUPERDOG_BUILD}.lib"
		HINTS
			"${GEMALTO_CUSTOMISED_SUPERDOG_LICENSING_API_DIR}"
			"${GEMALTO_SUPERDOG_LICENSING_API_DIR}"
		NO_CACHE
		REQUIRED
		NO_DEFAULT_PATH
	)
else ()
	message(FATAL_ERROR "Gemalto wrapper not supported on ${CMAKE_HOST_SYSTEM_NAME}.")
endif ()

add_library(gemalto_superdog_api STATIC IMPORTED)
set_property(
	TARGET
		gemalto_superdog_api
	PROPERTY IMPORTED_LOCATION "${GEMALTO_SUPERDOG_LICENSING_API_LIBRARY}"
)
set_property(
	TARGET
		gemalto_superdog_api
	PROPERTY SYSTEM TRUE
)
target_include_directories(gemalto_superdog_api INTERFACE
	"${GEMALTO_CUSTOMISED_SUPERDOG_LICENSING_API_DIR}"
	"${GEMALTO_SUPERDOG_LICENSING_API_DIR}"
)
target_link_libraries(gemalto_superdog_api INTERFACE
	legacy_stdio_definitions.lib
)

cmake_path(
	APPEND CMAKE_CURRENT_SOURCE_DIR
		"vendor_code.py"
	OUTPUT_VARIABLE GEMALTO_VENDOR_CODE_GENERATOR
)
cmake_path(
	APPEND GEMALTO_SUPERDOG_BASE
		"VendorCodes"
		"${GEMALTO_SUPERDOG_VENDOR_CODE_LABEL}.hvc"
	OUTPUT_VARIABLE GEMALTO_SUPERDOR_VENDOR_CODE_FILE
)
cmake_path(
	APPEND CMAKE_CURRENT_SOURCE_DIR
		"vendor_codes"
	OUTPUT_VARIABLE GEMALTO_VENDOR_CODE_CPP_DIR
)
cmake_path(
	APPEND GEMALTO_VENDOR_CODE_CPP_DIR
		"superdog.vc"
	OUTPUT_VARIABLE GEMALTO_SUPERDOR_VENDOR_CODE_CPP
)
file(MAKE_DIRECTORY "${GEMALTO_VENDOR_CODE_CPP_DIR}")
list(APPEND GEMALTO_SUPERDOG_VENDOR_CODE_GENERATOR
	"${Python_EXECUTABLE}"
	"${GEMALTO_VENDOR_CODE_GENERATOR}"
	"--name" "SUPERDOG"
	"--input-file" "${GEMALTO_SUPERDOR_VENDOR_CODE_FILE}"
	"--output-file" "${GEMALTO_SUPERDOR_VENDOR_CODE_CPP}"
)
execute_process(
	COMMAND
		${GEMALTO_SUPERDOG_VENDOR_CODE_GENERATOR}
	WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
	ENCODING UTF-8
	COMMAND_ERROR_IS_FATAL ANY
)
add_custom_command(
	OUTPUT
		"${GEMALTO_SUPERDOR_VENDOR_CODE_CPP}"
	COMMAND
		${GEMALTO_SUPERDOG_VENDOR_CODE_GENERATOR}
	DEPENDS
		"${GEMALTO_SUPERDOR_VENDOR_CODE_FILE}"
		"${GEMALTO_VENDOR_CODE_GENERATOR}"
	WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
	COMMENT "Generating Gemalto vendor code for SuperDog"
	VERBATIM
	USES_TERMINAL
)
add_custom_target(gemalto_superdog_vendor_code
	DEPENDS
		"${GEMALTO_SUPERDOR_VENDOR_CODE_CPP}"
)

if (MSVC)
	create_msvc_cxx_module_target(gemalto
		INCLUDE_DIRECTORIES
			"${CMAKE_CURRENT_SOURCE_DIR}/include"
		LINK_LIBRARIES
			gemalto_superdog_api
	)
	add_dependencies(gemalto gemalto_superdog_vendor_code)
	add_library(gemalto::gemalto ALIAS gemalto)
else ()
	message(FATAL_ERROR "Gemalto wrapper requires C++20 module support")
endif ()

add_executable(gemalto_driver EXCLUDE_FROM_ALL main.cpp)
target_link_libraries(gemalto_driver
	PRIVATE
		gemalto::gemalto
)
