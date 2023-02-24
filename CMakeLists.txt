cmake_minimum_required(VERSION 3.24)

set(__G_PROJECT_NAME "cpp-utilities")

project(${__G_PROJECT_NAME}
	VERSION 0.0
	LANGUAGES
		CXX
)

string(TOUPPER "${CMAKE_BUILD_TYPE}" __G_BUILD_TYPE_UPPER)

include(FetchContent)

macro (project_message LEVEL)
	message("${LEVEL}" "[${__G_PROJECT_NAME}] -- " ${ARGN})
endmacro ()

set(__G_OPTION_PREFIX "UTILITIES")

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(__G_LEGACY_ARCHITECTURE ON)
elseif (CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(__G_LEGACY_ARCHITECTURE OFF)
else ()
	project_message(FATAL_ERROR "Unknown pointer size: ${CMAKE_SIZEOF_VOID_P}")
endif ()

include("cmake/utils.cmake" NO_POLICY_SCOPE)

prefixed_option(FETCHCONTENT_BASE_DIR "" "The FETCHCONTENT_BASE_DIR variable for the current project."
	REFERENCE
	TYPE PATH
)
prefixed_option(FETCHCONTENT_GIT_SHALLOW ON "Fetch git repository as a shallow clone.")
prefixed_option(FETCHCONTENT_GIT_USE_SSH OFF "Use SSH to fetch git repository.")
prefixed_option(FETCHCONTENT_GITHUB_USE_SSH "FETCHCONTENT_GIT_USE_SSH" "Use SSH to fetch GitHub repository."
	REFERENCE
	PREFIX_REFERENCE
)
prefixed_option(FETCHCONTENT_GITLAB_USE_SSH "FETCHCONTENT_GIT_USE_SSH" "Use SSH to fetch GitLab repository."
	REFERENCE
	PREFIX_REFERENCE
)

prefixed_option(BUILD ON "Export all static library targets."
	EXTRA_PREFIXES
		"ALL"
)
prefixed_option(BUILD_EXAMPLES "" "Export all sample program targets."
	REFERENCE
	EXTRA_PREFIXES
		"ALL"
)
prefixed_option(BUILD_SHARED_LIBS "" "Export all shared library targets."
	REFERENCE
	EXTRA_PREFIXES
		"ALL"
)
prefixed_option(BUILD_TESTING "" "Export all shared library targets."
	REFERENCE
	EXTRA_PREFIXES
		"ALL"
)

get_prefixed_option(FETCHCONTENT_BASE_DIR)
get_prefixed_option(FETCHCONTENT_GIT_SHALLOW OUTPUT_PREFIXES "__G")
get_prefixed_option(FETCHCONTENT_GITHUB_USE_SSH OUTPUT_PREFIXES "__G")
get_prefixed_option(FETCHCONTENT_GITLAB_USE_SSH OUTPUT_PREFIXES "__G")

if (__G_FETCHCONTENT_GITHUB_USE_SSH)
	set(__C_GITHUB_PREFIX "git@github.com:" CACHE INTERNAL "" FORCE)
else ()
	set(__C_GITHUB_PREFIX "https://github.com/" CACHE INTERNAL "" FORCE)
endif ()

if (__G_FETCHCONTENT_GITLAB_USE_SSH)
	set(__C_GITLAB_PREFIX "git@gitlab.com:" CACHE INTERNAL "" FORCE)
else ()
	set(__C_GITLAB_PREFIX "https://gitlab.com/" CACHE INTERNAL "" FORCE)
endif ()

set(__G_GENERATED_FILES_DIR "${CMAKE_BINARY_DIR}/_gen")

set(__C_ARGPARSE_VERSION "2.9" CACHE INTERNAL "" FORCE)
set(__C_BOOST_VERSION "1.81.0" CACHE INTERNAL "" FORCE)
set(__C_CTRE_VERSION "3.7.2" CACHE INTERNAL "" FORCE)
set(__C_FMT_VERSION "9.1.0" CACHE INTERNAL "" FORCE)
set(__C_FREETYPE_VERSION "2.13.0" CACHE INTERNAL "" FORCE)
set(__C_HARFBUZZ_VERSION "7.0.1" CACHE INTERNAL "" FORCE)
set(__C_NLOHMANN_JSON_VERSION "3.11.2" CACHE INTERNAL "" FORCE)
set(__C_MIMALLOC_VERSION "2.0.9" CACHE INTERNAL "" FORCE)
set(__C_OPENCV_VERSION "4.7.0" CACHE INTERNAL "" FORCE)
set(__C_PNG_VERSION "1.6.39" CACHE INTERNAL "" FORCE)
set(__C_SPDLOG_VERSION "1.11.0" CACHE INTERNAL "" FORCE)
set(__C_XLNT_VERSION "1.5.0" CACHE INTERNAL "" FORCE)
set(__C_XLSXWRITER_VERSION "1.1.5" CACHE INTERNAL "" FORCE)
set(__C_ZLIB_VERSION "1.2.13" CACHE INTERNAL "" FORCE)

foreach (__L_DEPENDENCY_NAME
	"argparse"
	"ctre"
	"fmt"
	"mimalloc"
	"nlohmann_json"
	"spdlog"

	"zlib"
	"png"
	"freetype"
	"harfbuzz"

	"libxlsxwriter"
	"xlnt"
	"opencv"
	"boost"

	"mvs"

	"paddle"
)
	add_subdirectory("cmake/deps/${__L_DEPENDENCY_NAME}" EXCLUDE_FROM_ALL)
endforeach ()
