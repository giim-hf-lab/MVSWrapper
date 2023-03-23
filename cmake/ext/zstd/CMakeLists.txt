get_ext_version(zstd __L_VERSION)

add_ext_dep(zstd "${__G_GITHUB_PREFIX}" "facebook/zstd" "${__L_VERSION}" "v${__L_VERSION}" "build/cmake" "CONFIG"
	"-D" "ZSTD_BUILD_PROGRAMS:BOOL=FALSE"
)
