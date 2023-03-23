get_ext_version(xlsxwriter __L_VERSION)

add_ext_dep(xlsxwriter "${__G_GITHUB_PREFIX}" "jmcnamara/libxlsxwriter" "" "RELEASE_${__L_VERSION}" "" ""
	"-D" "ZLIB_USE_STATIC_LIBS:BOOL=TRUE"
	"-D" "USE_STANDARD_TMPFILE:BOOL=TRUE"
)

find_library(__L_LIBRARY xlsxwriter
	PATHS
		"${CMAKE_INSTALL_PREFIX}/lib"
	REQUIRED
	NO_DEFAULT_PATH
)

add_library(xlsxwriter::xlsxwriter STATIC IMPORTED GLOBAL)
set_property(
	TARGET
		xlsxwriter::xlsxwriter
	PROPERTY IMPORTED_LOCATION "${__L_LIBRARY}"
)
target_include_directories(xlsxwriter::xlsxwriter
	INTERFACE
		"${CMAKE_INSTALL_PREFIX}/include"
)
target_link_libraries(xlsxwriter::xlsxwriter
	INTERFACE
		ZLIB::ZLIB
)
