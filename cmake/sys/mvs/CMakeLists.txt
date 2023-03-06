if (NOT WIN32)
	project_message(FATAL_ERROR "MVS is not supported on non-Windows platforms.")
endif ()

prefixed_option(MVS_DEVELOPMENT_DIR "" "Set custom MVS development directory."
	REFERENCE
	TYPE STRING
)
prefixed_option(MVS_RUNTIME_DIR "" "Set custom MVS runtime directory."
	REFERENCE
	TYPE STRING
)

get_prefixed_option(MVS_DEVELOPMENT_DIR OUTPUT_PREFIXES "__L")
get_prefixed_option(MVS_RUNTIME_DIR OUTPUT_PREFIXES "__L")

if (NOT __L_MVS_DEVELOPMENT_DIR AND DEFINED ENV{MVCAM_COMMON_RUNENV})
	set(__L_MVS_DEVELOPMENT_DIR "$ENV{MVCAM_COMMON_RUNENV}")
endif ()

if (DEFINED ENV{PROGRAMFILES\(x86\)})
	if (NOT __L_MVS_DEVELOPMENT_DIR)
		set(__L_MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES\(x86\)}/MVS/Development")
	endif ()
	if (NOT __L_MVS_RUNTIME_DIR)
		set(__L_MVS_RUNTIME_DIR "$ENV{PROGRAMFILES\(x86\)}/Common Files/MVS/Runtime")
	endif ()
elseif (DEFINED ENV{PROGRAMFILES})
	if (NOT __L_MVS_DEVELOPMENT_DIR)
		set(__L_MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES}/MVS/Development")
	endif ()
	if (NOT __L_MVS_RUNTIME_DIR)
		set(__L_MVS_RUNTIME_DIR "$ENV{PROGRAMFILES}/Common Files/MVS/Runtime")
	endif ()
else ()
	project_message(FATAL_ERROR "Unknown Windows environment detected.")
endif ()

set(__L_MVS_INCLUDE_DIR "${__L_MVS_DEVELOPMENT_DIR}/Includes")
if (__G_LEGACY_ARCHITECTURE)
	set(__L_MVS_DLL_DIR "${__L_MVS_RUNTIME_DIR}/Win32_i86")
	set(__L_MVS_LIBRARY_DIR "${__L_MVS_DEVELOPMENT_DIR}/Libraries/win32")
else ()
	set(__L_MVS_DLL_DIR "${__L_MVS_RUNTIME_DIR}/Win64_x64")
	set(__L_MVS_LIBRARY_DIR "${__L_MVS_DEVELOPMENT_DIR}/Libraries/win64")
endif ()

add_library(mvs::mvs SHARED IMPORTED GLOBAL)
set_property(
	TARGET mvs::mvs
	PROPERTY IMPORTED_LOCATION "${__L_MVS_DLL_DIR}/MvCameraControl.dll"
)
set_property(
	TARGET mvs::mvs
	PROPERTY IMPORTED_IMPLIB "${__L_MVS_LIBRARY_DIR}/MvCameraControl.lib"
)
set_property(
	TARGET mvs::mvs
	PROPERTY INTERFACE_INCLUDE_DIRECTORIES
		"${__L_MVS_INCLUDE_DIR}"
)

set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "mvs::mvs")
