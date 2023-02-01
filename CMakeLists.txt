cmake_minimum_required(VERSION 3.24)

set(CURRENT_PROJECT_NAME "cpp-utilities")

project(${CURRENT_PROJECT_NAME}
	VERSION 0.0
	LANGUAGES
		CXX
)

include(CMakeDependentOption)

find_package(Git 2.22.0)

set(PROJECT_OPTIONS_PREFIX "UTILITIES")
set(PROJECT_DEFINED_COMPONENTS)

# [Start] Options

cmake_dependent_option(${PROJECT_OPTIONS_PREFIX}_ENSURE_SUBMODULES "Ensure submodules have been initialised." OFF "GIT_FOUND" OFF)
option(${PROJECT_OPTIONS_PREFIX}_USE_MIMALLOC "Use mimalloc as the default allocator." ON)
cmake_dependent_option(${PROJECT_OPTIONS_PREFIX}_USE_SECURE_MIMALLOC "Use secure mimalloc." OFF "${PROJECT_OPTIONS_PREFIX}_USE_MIMALLOC" OFF)

option(${PROJECT_OPTIONS_PREFIX}_DEPENDENCIES_FAIL_FAST "Fail immediately after encoutering a non-existing submodule." ON)
option(${PROJECT_OPTIONS_PREFIX}_BUILD_ALL "All libraries in this project will be built." ON)
option(${PROJECT_OPTIONS_PREFIX}_BUILD_SHARED "Build shared library for each components (<COMPONENT_NAME>-shared suffix)." $<BOOL:${BUILD_SHARED_LIBS}>)

include("cmake/options.cmake" NO_POLICY_SCOPE)

set(${PROJECT_OPTIONS_PREFIX}_EXPORT_DEPENDENCIES "" CACHE STRING "Extra dependencies to be exported.")

# [End] Options

set(INCLUDED_DEPENDENCIES "argparse" "fmt" "spdlog")

if (${PROJECT_OPTIONS_PREFIX}_USE_MIMALLOC)
	list(APPEND INCLUDED_DEPENDENCIES "mimalloc")
endif ()

foreach (COMPONENT_NAME IN LISTS PROJECT_DEFINED_COMPONENTS)
	set(COMPONENT_DEPS_CMAKE "cmake/comp/${COMPONENT_NAME}.deps.cmake")
	if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${COMPONENT_DEPS_CMAKE}")
		include("${COMPONENT_DEPS_CMAKE}" NO_POLICY_SCOPE)
	endif ()
endforeach ()

list(APPEND INCLUDED_DEPENDENCIES ${${PROJECT_OPTIONS_PREFIX}_EXPORT_DEPENDENCIES})

list(REMOVE_DUPLICATES INCLUDED_DEPENDENCIES)
set(INCLUDED_SUBMODULES)
foreach (DEPENDENCY IN LISTS INCLUDED_DEPENDENCIES)
	set(SUBMODULE "ext/${DEPENDENCY}")
	if (IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${SUBMODULE}")
		list(APPEND INCLUDED_SUBMODULES "${SUBMODULE}")
	else ()
		set(ERROR_MESSAGE "[${CURRENT_PROJECT_NAME}] Dependency ${DEPENDENCY} is not a submodule.")
		if (${PROJECT_OPTIONS_PREFIX}_DEPENDENCIES_FAIL_FAST)
			message(FATAL_ERROR "${ERROR_MESSAGE}")
		else ()
			message(WARNING "${ERROR_MESSAGE} Ignoring...")
		endif ()
	endif ()
endforeach ()

string(JOIN ", " INCLUDED_SUBMODULES_STRING ${INCLUDED_SUBMODULES})
message(STATUS "[${CURRENT_PROJECT_NAME}] Build with the following submodules: ${INCLUDED_SUBMODULES_STRING}")
if (UTILITIES_ENSURE_SUBMODULES)
	message(STATUS "[${CURRENT_PROJECT_NAME}] Running submodule update as requested...")
	execute_process(
		COMMAND ${GIT_EXECUTABLE}
			"submodule"
			"update"
			"--force"
			"--init"
			"--recursive"
			${INCLUDED_SUBMODULES}
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
		COMMAND_ECHO STDOUT
		COMMAND_ERROR_IS_FATAL ANY
	)
endif ()
foreach (SUBMODULE IN LISTS INCLUDED_SUBMODULES)
	set(SUBMODULE_CMAKE "cmake/${SUBMODULE}.cmake")
	if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUBMODULE_CMAKE}")
		include("${SUBMODULE_CMAKE}" NO_POLICY_SCOPE)
	else ()
		add_subdirectory("${SUBMODULE}" EXCLUDE_FROM_ALL)
	endif ()
endforeach ()

include("cmake/cxx_module.cmake" NO_POLICY_SCOPE)

set(PROJECT_ALIAS_NAMESPACE "utilities")

add_subdirectory("comp" EXCLUDE_FROM_ALL)
