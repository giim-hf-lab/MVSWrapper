if (HIKVISION_MVS_FOUND)
	return ()
endif ()

if (HikVisionMVS_FIND_REQUIRED)
	set(NOT_FOUND_LEVEL FATAL_ERROR)
else ()
	set(NOT_FOUND_LEVEL WARNING)
endif ()

if (NOT WIN32)
	message(${NOT_FOUND_LEVEL} "FindHikVisionMVS not supported on non-Windows platforms by now.")
	return ()
endif ()

if (DEFINED ENV{PROGRAMFILES\(x86\)})
	set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES\(x86\)}/MVS/Development")
	set(MVS_RUNTIME_DIR "$ENV{PROGRAMFILES\(x86\)}/Common Files/MVS/Runtime")
elseif (DEFINED ENV{PROGRAMFILES})
	set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES}/MVS/Development")
	set(MVS_RUNTIME_DIR "$ENV{PROGRAMFILES}/Common Files/MVS/Runtime")
else ()
	message(FATAL_ERROR "Unknown Windows environment detected.")
endif ()

set(MVS_INCLUDE_DIR "${MVS_DEVELOPMENT_DIR}/Includes")
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(MVS_DLL_DIR "${MVS_RUNTIME_DIR}/Win64_x64")
	set(MVS_LIBRARY_DIR "${MVS_DEVELOPMENT_DIR}/Libraries/win64")
elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(MVS_DLL_DIR "${MVS_RUNTIME_DIR}/Win32_i86")
	set(MVS_LIBRARY_DIR "${MVS_DEVELOPMENT_DIR}/Libraries/win32")
else ()
	message(FATAL_ERROR "Unknown Windows architecture.")
endif ()

add_library(hikvision::mvs SHARED IMPORTED GLOBAL)
set_property(
	TARGET hikvision::mvs
	PROPERTY IMPORTED_LOCATION "${MVS_DLL_DIR}/MvCameraControl.dll"
)
set_property(
	TARGET hikvision::mvs
	PROPERTY IMPORTED_IMPLIB "${MVS_LIBRARY_DIR}/MvCameraControl.lib"
)
set_property(
	TARGET hikvision::mvs
	PROPERTY INTERFACE_INCLUDE_DIRECTORIES
		"${MVS_INCLUDE_DIR}"
)
