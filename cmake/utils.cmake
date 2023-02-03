include_guard(DIRECTORY)

function (check_conditions NAME)
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

	set(__L_NAME "__C_${NAME}_CHECKED")
	foreach (__L_CONDITION ${__P_CONDITIONS})
		if (${__L_INVERT} ${__P_TYPE} ${__L_CONDITION})
			set("${__L_NAME}" "${__L_SHORT_VALUE}" CACHE INTERNAL "")
			return ()
		endif ()
	endforeach ()
	set("${__L_NAME}" "${__L_FINAL_VALUE}" CACHE INTERNAL "")
endfunction ()

function (prefixed_option NAME VALUE COMMENT)
	cmake_parse_arguments(
		"__P"
		"ADVANCED;PREFIX_REFERENCE;REFERENCE;REQUIRED;UNAVAILABLE_WARNING"
		"DEFAULT;TYPE"
		"DEPENDS;EXTRA_PREFIXES;REFERENCE_EXTRA_PREFIXES"
		${ARGN}
	)

	string(JOIN "_" __L_OPTION_NAME "${__G_OPTION_PREFIX}" ${__P_EXTRA_PREFIXES} "${NAME}")

	check_conditions("${__L_OPTION_NAME}" ALL CONDITIONS ${__P_DEPENDS})
	if ($CACHE{__C_${__L_OPTION_NAME}_CHECKED})
		if (__P_REFERENCE)
			if (__P_PREFIX_REFERENCE)
				set(__L_VALUE_NAME_FRAGMENTS "${__G_OPTION_PREFIX}")
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
	string(JOIN "_" __L_OPTION_NAME "${__G_OPTION_PREFIX}" ${__P_EXTRA_PREFIXES} "${NAME}")
	string(JOIN "_" __P_OUTPUT_VARIABLE ${__P_OUTPUT_PREFIXES} "${__P_OUTPUT_VARIABLE}")
	set("${__P_OUTPUT_VARIABLE}" ${${__L_OPTION_NAME}} PARENT_SCOPE)
endfunction ()
