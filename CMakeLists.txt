cmake_minimum_required(VERSION 3.24)

set(__G_PROJECT_NAME "cpp-utilities")

project(${__G_PROJECT_NAME}
	VERSION 0.0
	LANGUAGES
		CXX
)

include(FetchContent)
include("cmake/utils.cmake" NO_POLICY_SCOPE)

enforce(CMAKE_POLICY_DEFAULT_CMP0054 NEW)
enforce(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
enforce(CMAKE_POLICY_DEFAULT_CMP0097 NEW)
enforce(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
enforce(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

enforce(CMAKE_C_STANDARD 17)
enforce(CMAKE_C_STANDARD_REQUIRED TRUE)
enforce(CMAKE_C_EXTENSIONS FALSE)
enforce(CMAKE_CXX_STANDARD 20)
enforce(CMAKE_CXX_STANDARD_REQUIRED TRUE)
enforce(CMAKE_CXX_EXTENSIONS FALSE)

enforce(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)
enforce(CMAKE_TLS_VERIFY TRUE)

string(TOUPPER "${CMAKE_BUILD_TYPE}" __G_BUILD_TYPE_UPPER)
set(__G_CACHE_PREFIX "UTILITIES")
set(__G_GENERATED_FILES_DIR "${CMAKE_BINARY_DIR}/_gen")
string(TOLOWER "${__G_CACHE_PREFIX}" __G_NAMESPACE)

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(__G_LEGACY_ARCHITECTURE ON)
elseif (CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(__G_LEGACY_ARCHITECTURE OFF)
else ()
	project_message(FATAL_ERROR "Unknown pointer size: ${CMAKE_SIZEOF_VOID_P}")
endif ()

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

get_prefixed_option(FETCHCONTENT_BASE_DIR)
get_prefixed_option(FETCHCONTENT_GIT_SHALLOW OUTPUT_PREFIXES "__G")
get_prefixed_option(FETCHCONTENT_GITHUB_USE_SSH OUTPUT_PREFIXES "__G")
get_prefixed_option(FETCHCONTENT_GITLAB_USE_SSH OUTPUT_PREFIXES "__G")

if (__G_FETCHCONTENT_GITHUB_USE_SSH)
	set(__G_GITHUB_PREFIX "git@github.com:")
else ()
	set(__G_GITHUB_PREFIX "https://github.com/")
endif ()

if (__G_FETCHCONTENT_GITLAB_USE_SSH)
	set(__G_GITLAB_PREFIX "git@gitlab.com:")
else ()
	set(__G_GITLAB_PREFIX "https://gitlab.com/")
endif ()

set_internal_cache(EXTERNAL_DEPENDENCIES FALSE)
set_ext_version(argparse "2.9")
set_ext_version(boost "1.81.0")
set_ext_version(ctre "3.7.2")
set_ext_version(fmt "9.1.0")
set_ext_version(freetype "2.13.0")
set_ext_version(harfbuzz "7.1.0")
set_ext_version(nameof "0.10.2")
set_ext_version(nlohmann_json "3.11.2")
set_ext_version(mimalloc "2.0.9")
set_ext_version(mini "0.9.14")
set_ext_version(opencv "4.7.0")
set_ext_version(paddle "2.4.2")
set_ext_version(png "1.6.39")
set_ext_version(spdlog "1.11.0")
set_ext_version(sqlite3 "3.41.1")
set_ext_version(xlnt "1.5.0")
set_ext_version(xlsxwriter "1.1.5")
set_ext_version(zlib "1.2.13")

prefixed_option(BUILD_ALL OFF "Enable all external dependencies.")
get_prefixed_option(BUILD_ALL OUTPUT_PREFIXES "__G")

set_internal_cache(EXPORTED_COMPONENTS FALSE)
foreach (__L_COMPONENT
	"camera"
	"ini"
	"preprocessor"
	"winutils"
)
	string(TOUPPER "${__L_COMPONENT}" __L_COMPONENT_UPPER)
	set(__L_GUARD_NAME "BUILD_${__L_COMPONENT_UPPER}")

	prefixed_option("${__L_GUARD_NAME}" "${__G_BUILD_ALL}" "Build `${__L_COMPONENT}`.")

	get_prefixed_option("${__L_GUARD_NAME}" OUTPUT_VARIABLE __L_COMPONENT_BUILT)
	if (__L_COMPONENT_BUILT)
		set_internal_cache(EXPORTED_COMPONENTS TRUE "${__L_COMPONENT}")
	endif ()
endforeach ()

get_internal_cache(EXPORTED_COMPONENTS __L_COMPONENTS)
set_internal_cache(EXPORTED_COMPONENTS FALSE)
prepare("comp" "" ${__L_COMPONENTS})

prefixed_option(ENABLE_ALL OFF "Enable all external dependencies.")
get_prefixed_option(ENABLE_ALL OUTPUT_PREFIXES "__G")

set_internal_cache(EXPORTED_EXTERNAL_DEPENDENCIES FALSE)
get_internal_cache(EXTERNAL_DEPENDENCIES __L_DEPENDENCIES)
foreach (__L_DEPENDENCY ${__L_DEPENDENCIES})
	string(TOUPPER "${__L_DEPENDENCY}" __L_DEPENDENCY_UPPER)
	set(__L_GUARD_NAME "EXPORT_${__L_DEPENDENCY_UPPER}")

	prefixed_option("${__L_GUARD_NAME}" "${__G_ENABLE_ALL}" "Fetch and export targets created by `${__L_DEPENDENCY}`.")

	get_prefixed_option("${__L_GUARD_NAME}" OUTPUT_VARIABLE __L_DEPENDENCY_EXPORTED)
	if (__L_DEPENDENCY_EXPORTED)
		set_internal_cache(EXPORTED_EXTERNAL_DEPENDENCIES TRUE "${__L_DEPENDENCY}")
	endif ()
endforeach ()

get_internal_cache(EXPORTED_EXTERNAL_DEPENDENCIES __L_DEPENDENCIES)
set_internal_cache(EXPORTED_EXTERNAL_DEPENDENCIES FALSE)
prepare("ext" "" ${__L_DEPENDENCIES})

get_internal_cache(EXPORTED_EXTERNAL_DEPENDENCIES __L_DEPENDENCIES)
list(REMOVE_DUPLICATES __L_DEPENDENCIES)
set_internal_cache(EXPORTED_EXTERNAL_DEPENDENCIES FALSE ${__L_DEPENDENCIES})

set_internal_cache(TARGETS_ENFORCE_SYSTEM FALSE)
foreach (__L_DEPENDENCY ${__L_DEPENDENCIES})
	add_subdirectory("cmake/ext/${__L_DEPENDENCY}" EXCLUDE_FROM_ALL)
endforeach ()

foreach (__L_DEPENDENCY "basler" "mvs")
	string(TOUPPER "${__L_DEPENDENCY}" __L_DEPENDENCY_UPPER)
	set(__L_GUARD_NAME "FIND_${__L_DEPENDENCY_UPPER}")

	prefixed_option("${__L_GUARD_NAME}" "${__G_ENABLE_ALL}" "Find and create targets for `${__L_DEPENDENCY}`.")

	get_prefixed_option("${__L_GUARD_NAME}" OUTPUT_VARIABLE __L_DEPENDENCY_ENABLED)
	if (__L_DEPENDENCY_ENABLED)
		add_subdirectory("cmake/sys/${__L_DEPENDENCY}" EXCLUDE_FROM_ALL)
	endif ()
endforeach ()

get_internal_cache(TARGETS_ENFORCE_SYSTEM __L_TARGETS)
set_property(
	TARGET
		${__L_TARGETS}
	PROPERTY SYSTEM TRUE
)

get_internal_cache(EXPORTED_COMPONENTS __L_COMPONENTS)
foreach (__L_COMPONENT ${__L_COMPONENTS})
	add_subdirectory("comp/${__L_COMPONENT}" EXCLUDE_FROM_ALL)
endforeach ()
