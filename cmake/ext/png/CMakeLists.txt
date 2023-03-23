get_ext_version(png __L_VERSION)

add_ext_dep(PNG "${__G_GITHUB_PREFIX}" "glennrp/libpng" "${__L_VERSION}" "v${__L_VERSION}" "" ""
	"-D" "ZLIB_USE_STATIC_LIBS:BOOL=TRUE"
	"-D" "PNG_SHARED:BOOL=FALSE"
	"-D" "PNG_STATIC:BOOL=TRUE"
	"-D" "PNG_EXECUTABLES:BOOL=FALSE"
	"-D" "PNG_TESTS:BOOL=FALSE"
)
