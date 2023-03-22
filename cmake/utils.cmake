include_guard(GLOBAL)

macro (project_message LEVEL)
	message("${LEVEL}" "[${__G_PROJECT_NAME}] -- " ${ARGN})
endmacro ()

macro (enforce NAME)
	set("${NAME}" ${ARGN})
	if (DEFINED CACHE{${NAME}})
		set_property(
			CACHE "${NAME}"
			PROPERTY VALUE
				${ARGN}
		)
		set_property(
			CACHE "${NAME}"
			PROPERTY TYPE
				INTERNAL
		)
	else ()
		set("${NAME}" ${ARGN} CACHE INTERNAL "" FORCE)
	endif ()
endmacro ()

macro (set_internal_cache NAME APPEND)
	set(__L_NAME "__C_${__G_CACHE_PREFIX}_INTERNAL_${NAME}")
	if (DEFINED CACHE{${__L_NAME}})
		if (${APPEND})
			set(__L_APPEND "APPEND")
		else ()
			set(__L_APPEND)
		endif ()
		set_property(
			CACHE "${__L_NAME}"
			${__L_APPEND}
			PROPERTY VALUE
				${ARGN}
		)
		set_property(
			CACHE "${__L_NAME}"
			PROPERTY TYPE
				INTERNAL
		)
	else ()
		set("${__L_NAME}" ${ARGN} CACHE INTERNAL "" FORCE)
	endif ()
endmacro ()

macro (get_internal_cache NAME OUTPUT_VARIABLE)
	get_property("${OUTPUT_VARIABLE}"
		CACHE "__C_${__G_CACHE_PREFIX}_INTERNAL_${NAME}"
		PROPERTY VALUE
	)
endmacro ()

function (check_conditions NAME OUTPUT_VARIABLE)
	cmake_parse_arguments(
		"__P"
		"ALL"
		"TYPE"
		"CONDITIONS"
		${ARGN}
	)

	if (__P_ALL)
		set(__L_INVERT NOT)
		set(__L_SHORT_VALUE FALSE)
		set(__L_FINAL_VALUE TRUE)
	else ()
		set(__L_INVERT)
		set(__L_SHORT_VALUE TRUE)
		set(__L_FINAL_VALUE FALSE)
	endif ()

	set(__L_NAME "${NAME}_CHECKED")
	foreach (__L_CONDITION ${__P_CONDITIONS})
		if (${__L_INVERT} ${__P_TYPE} ${__L_CONDITION})
			set("${OUTPUT_VARIABLE}" ${__L_SHORT_VALUE} PARENT_SCOPE)
			return ()
		endif ()
	endforeach ()
	set("${OUTPUT_VARIABLE}" ${__L_FINAL_VALUE} PARENT_SCOPE)
endfunction ()

function (prefixed_option NAME VALUE COMMENT)
	cmake_parse_arguments(
		"__P"
		"ADVANCED;PREFIX_REFERENCE;REFERENCE;REQUIRED;UNAVAILABLE_WARNING"
		"DEFAULT;TYPE"
		"DEPENDS;EXTRA_PREFIXES;REFERENCE_EXTRA_PREFIXES"
		${ARGN}
	)

	string(JOIN "_" __L_OPTION_NAME "${__G_CACHE_PREFIX}" ${__P_EXTRA_PREFIXES} "${NAME}")

	check_conditions("${__L_OPTION_NAME}" __L_CHECKED ALL CONDITIONS ${__P_DEPENDS})
	if (__L_CHECKED)
		if (__P_REFERENCE)
			if (__P_PREFIX_REFERENCE)
				set(__L_VALUE_NAME_FRAGMENTS "${__G_CACHE_PREFIX}")
			else ()
				set(__L_VALUE_NAME_FRAGMENTS)
			endif ()
			list(APPEND __L_VALUE_NAME_FRAGMENTS ${__P_REFERENCE_EXTRA_PREFIXES})
			if ("${VALUE}" STREQUAL "")
				list(APPEND __L_VALUE_NAME_FRAGMENTS "${NAME}")
			else ()
				list(APPEND __L_VALUE_NAME_FRAGMENTS "${VALUE}")
			endif ()
			string(JOIN "_" __L_VALUE ${__L_VALUE_NAME_FRAGMENTS})
			set(__L_VALUE ${${__L_VALUE}})
		else ()
			set(__L_VALUE ${VALUE})
		endif ()
		if ("${__P_TYPE}" STREQUAL "")
			set(__P_TYPE "BOOL")
		endif ()
		if ("${__P_TYPE}" STREQUAL "BOOL")
			if (__L_VALUE)
				set(__L_VALUE TRUE)
			else ()
				set(__L_VALUE FALSE)
			endif ()
		endif ()
	else ()
		set(__L_VALUE ${__P_DEFAULT})
		set(__P_ADVANCED OFF)
		set(__P_TYPE "INTERNAL")
		unset("${__L_OPTION_NAME}" PARENT_SCOPE)
		if (__P_UNAVAILABLE_WARNING)
			project_message(WARNING "Ignoring unavailable option ${__L_OPTION_NAME}.")
		endif ()
	endif ()

	set("${__L_OPTION_NAME}" "${__L_VALUE}" CACHE "${__P_TYPE}" "${COMMENT}" FORCE)
	if (__P_ADVANCED)
		mark_as_advanced(FORCE "${__L_OPTION_NAME}")
	endif ()
endfunction ()

macro (set_option NAME)
	set("${__G_CACHE_PREFIX}_${NAME}" ${ARGN})
endmacro ()

function (get_prefixed_option NAME)
	cmake_parse_arguments(
		"__P"
		""
		"OUTPUT_VARIABLE"
		"EXTRA_PREFIXES;OUTPUT_PREFIXES"
		${ARGN}
	)
	if (NOT __P_OUTPUT_VARIABLE)
		set(__P_OUTPUT_VARIABLE "${NAME}")
	endif ()
	string(JOIN "_" __L_OPTION_NAME "${__G_CACHE_PREFIX}" ${__P_EXTRA_PREFIXES} "${NAME}")
	string(JOIN "_" __P_OUTPUT_VARIABLE ${__P_OUTPUT_PREFIXES} "${__P_OUTPUT_VARIABLE}")
	set("${__P_OUTPUT_VARIABLE}" ${${__L_OPTION_NAME}} PARENT_SCOPE)
endfunction ()

macro (set_ext_version NAME VERSION)
	string(TOUPPER "${NAME}" __L_NAME_UPPER)
	set_internal_cache("VERSION_${__L_NAME_UPPER}" FALSE "${VERSION}")
	set_internal_cache(EXTERNAL_DEPENDENCIES TRUE "${NAME}")
endmacro ()

macro (get_ext_version NAME OUTPUT_VARIABLE)
	string(TOUPPER "${NAME}" __L_NAME_UPPER)
	get_internal_cache("VERSION_${__L_NAME_UPPER}" __L_VERSION)
	set("${OUTPUT_VARIABLE}" "${__L_VERSION}")
endmacro ()

macro (prepare TYPE PREFIX)
	foreach (__L_NAME ${ARGN})
		include("cmake/pre/${TYPE}/${PREFIX}${__L_NAME}.cmake" NO_POLICY_SCOPE)
	endforeach ()
endmacro ()

macro (try_get_property TARGET OUTPUT_VARIABLE VALUE)
	set("${OUTPUT_VARIABLE}" "${VALUE}")
	foreach (__L_PROPERTY ${ARGN})
		get_property(__L_VALUE
			TARGET "${TARGET}"
			PROPERTY "${__L_PROPERTY}"
		)
		if (NOT "${__L_VALUE}" STREQUAL "")
			set("${OUTPUT_VARIABLE}" "${__L_VALUE}")
			break ()
		endif ()
	endforeach ()
endmacro ()

macro (get_archive_path TARGET BINARY_DIR OUTPUT_VARIABLE)
	try_get_property("${TARGET}" __L_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
		"ARCHIVE_OUTPUT_DIRECTORY_${__G_BUILD_TYPE_UPPER}"
		"ARCHIVE_OUTPUT_DIRECTORY"
	)
	if ("${__L_OUTPUT_DIRECTORY}" STREQUAL "")
		set(__L_OUTPUT_DIRECTORY "${BINARY_DIR}")
	endif ()

	try_get_property("${TARGET}" __L_PREFIX "${CMAKE_STATIC_LIBRARY_PREFIX}"
		"PREFIX"
	)

	try_get_property("${TARGET}" __L_SUFFIX "${CMAKE_STATIC_LIBRARY_SUFFIX}"
		"SUFFIX"
	)

	try_get_property("${TARGET}" __L_OUTPUT_NAME "${TARGET}"
		"ARCHIVE_OUTPUT_NAME_${__G_BUILD_TYPE_UPPER}"
		"ARCHIVE_OUTPUT_NAME"
		"OUTPUT_NAME_${__G_BUILD_TYPE_UPPER}"
		"OUTPUT_NAME"
	)

	try_get_property("${TARGET}" __L_POSTFIX "" "${__G_BUILD_TYPE_UPPER}_POSTFIX")

	set("${OUTPUT_VARIABLE}" "${__L_OUTPUT_DIRECTORY}/${__L_PREFIX}${__L_OUTPUT_NAME}${__L_POSTFIX}${__L_SUFFIX}")
endmacro ()

macro (ext_compile_install NAME LANGUAGE SOURCE_DIR SUBDIR BINARY_DIR)
	set(__L_CONFIG_ARGS
		"-S" "${SOURCE_DIR}${SUBDIR}"
		"-B" "${BINARY_DIR}"
		"-D" "CMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
	)
	foreach (__L_LANGUAGE ${LANGUAGE})
		list(APPEND __L_CONFIG_ARGS
			"-D" "CMAKE_${__L_LANGUAGE}_COMPILER:STRING=${CMAKE_${__L_LANGUAGE}_COMPILER}"
			"-D" "CMAKE_${__L_LANGUAGE}_EXTENSIONS:BOOL=${CMAKE_${__L_LANGUAGE}_EXTENSIONS}"
			"-D" "CMAKE_${__L_LANGUAGE}_FLAGS_INIT:STRING=${CMAKE_${__L_LANGUAGE}_FLAGS_INIT}"
			"-D" "CMAKE_${__L_LANGUAGE}_FLAGS_DEBUG_INIT:STRING=${CMAKE_${__L_LANGUAGE}_FLAGS_DEBUG_INIT}"
			"-D" "CMAKE_${__L_LANGUAGE}_FLAGS_RELEASE_INIT:STRING=${CMAKE_${__L_LANGUAGE}_FLAGS_RELEASE_INIT}"
			"-D" "CMAKE_${__L_LANGUAGE}_STANDARD:STRING=${CMAKE_${__L_LANGUAGE}_STANDARD}"
			"-D" "CMAKE_${__L_LANGUAGE}_STANDARD_REQUIRED:BOOL=${CMAKE_${__L_LANGUAGE}_STANDARD_REQUIRED}"
		)
	endforeach ()
	list(APPEND __L_CONFIG_ARGS
		"-D" "CMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}"
		"-D" "CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT:BOOL=FALSE"
	)
	if (MSVC)
		list(APPEND __L_CONFIG_ARGS
			"-D" "CMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}"
		)
	endif ()

	execute_process(
		COMMAND "${CMAKE_COMMAND}"
			${__L_CONFIG_ARGS}
			${ARGN}
			"-G" "${CMAKE_GENERATOR}"
			"-T" "${CMAKE_GENERATOR_TOOLSET}"
			"-A" "${CMAKE_GENERATOR_PLATFORM}"
			"--toolchain" "${CMAKE_TOOLCHAIN_FILE}"
			"--install-prefix" "${CMAKE_INSTALL_PREFIX}"
			"-Wno-dev"
		WORKING_DIRECTORY "${SOURCE_DIR}"
		OUTPUT_FILE "${BINARY_DIR}/${NAME}.config.out"
		ERROR_FILE "${BINARY_DIR}/${NAME}.config.err"
		COMMAND_ECHO "STDOUT"
		ENCODING "UTF-8"
		COMMAND_ERROR_IS_FATAL "ANY"
	)
	execute_process(
		COMMAND "${CMAKE_COMMAND}"
			"--build" "${BINARY_DIR}"
		WORKING_DIRECTORY "${SOURCE_DIR}"
		OUTPUT_FILE "${BINARY_DIR}/${NAME}.build.out"
		ERROR_FILE "${BINARY_DIR}/${NAME}.build.err"
		COMMAND_ECHO "STDOUT"
		ENCODING "UTF-8"
		COMMAND_ERROR_IS_FATAL "ANY"
	)
	execute_process(
		COMMAND "${CMAKE_COMMAND}"
			"--install" "${BINARY_DIR}"
			"--prefix" "${CMAKE_INSTALL_PREFIX}"
		WORKING_DIRECTORY "${SOURCE_DIR}"
		OUTPUT_FILE "${BINARY_DIR}/${NAME}.install.out"
		ERROR_FILE "${BINARY_DIR}/${NAME}.install.err"
		COMMAND_ECHO "STDOUT"
		ENCODING "UTF-8"
		COMMAND_ERROR_IS_FATAL "ANY"
	)
endmacro ()

macro (add_ext_dep NAME PREFIX SUFFIX VERSION TAG LANGUAGE SUBDIR)
	FetchContent_Declare("${NAME}"
		GIT_REPOSITORY "${PREFIX}${SUFFIX}.git"
		GIT_TAG "${TAG}"
		GIT_SUBMODULES_RECURSE TRUE
		GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
		GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
	)
	FetchContent_Populate("${NAME}")
	FetchContent_GetProperties("${NAME}" SOURCE_DIR __L_SOURCE_DIR BINARY_DIR __L_BINARY_DIR)

	ext_compile_install("${NAME}" "${LANGUAGE}" "${__L_SOURCE_DIR}" "${SUBDIR}" "${__L_BINARY_DIR}" ${ARGN})

	if (NOT "${VERSION}" STREQUAL "")
		find_package("${NAME}" "${VERSION}" REQUIRED GLOBAL BYPASS_PROVIDER)
	endif ()
endmacro ()

macro (create_library_target NAME PREFIX OUTPUT_VARIABLE)
	set(__L_CANONICAL_NAME "${PREFIX}_${NAME}")
	set(__L_NAMESPACE_NAME "${PREFIX}::${NAME}")
	add_library("${__L_CANONICAL_NAME}" ${ARGN})
	add_library("${__L_NAMESPACE_NAME}" ALIAS "${__L_CANONICAL_NAME}")

	set("${OUTPUT_VARIABLE}" "${__L_CANONICAL_NAME}")
endmacro ()

macro (create_example NAME PREFIX OUTPUT_VARIABLE)
	set(__L_CANONICAL_NAME "${PREFIX}_${NAME}_example")
	add_executable("${__L_CANONICAL_NAME}"  ${ARGN})

	set("${OUTPUT_VARIABLE}" "${__L_CANONICAL_NAME}")
endmacro ()
