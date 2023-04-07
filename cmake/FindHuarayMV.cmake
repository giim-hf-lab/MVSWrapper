if (HUARAY_MV_FOUND)
	return ()
endif ()

if (HuarayMV_FIND_REQUIRED)
	set(NOT_FOUND_LEVEL FATAL_ERROR)
else ()
	set(NOT_FOUND_LEVEL WARNING)
endif ()

if (NOT WIN32)
	message(${NOT_FOUND_LEVEL} "FindHuarayMV not supported on non-Windows platforms by now.")
	return ()
endif ()

set(MV_DEVELOPMENT_DIR "$ENV{PROGRAMFILES}/HuarayTech/MV Viewer/Development")
set(MV_RUNTIME_DIR "$ENV{PROGRAMFILES}/HuarayTech/MV Viewer/Runtime")

set(MV_INCLUDE_DIR "${MV_DEVELOPMENT_DIR}/Include")
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(MV_DLL_DIR "${MV_RUNTIME_DIR}/x64")
	set(MV_LIBRARY_DIR "${MV_DEVELOPMENT_DIR}/Lib/x64")
elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(MV_DLL_DIR "${MV_RUNTIME_DIR}/Win32")
	set(MV_LIBRARY_DIR "${MV_DEVELOPMENT_DIR}/Lib/win32")
else ()
	message(FATAL_ERROR "Unknown Windows architecture.")
endif ()

add_library(huaray::mv SHARED IMPORTED GLOBAL)
set_property(
	TARGET huaray::mv
	PROPERTY IMPORTED_LOCATION "${MV_LIBRARY_DIR}/MVSDKmd.dll"
)
set_property(
	TARGET huaray::mv
	PROPERTY IMPORTED_IMPLIB "${MV_LIBRARY_DIR}/MVSDKmd.lib"
)
set_property(
	TARGET huaray::mv
	PROPERTY INTERFACE_INCLUDE_DIRECTORIES
		"${MV_INCLUDE_DIR}"
)
