get_ext_version(mimalloc __L_VERSION)

add_ext_dep(mimalloc "${__G_GITHUB_PREFIX}" "microsoft/mimalloc" "${__L_VERSION}" "v${__L_VERSION}" "C;CXX" ""
	"-D" "MI_USE_CXX:BOOL=TRUE"
	"-D" "MI_BUILD_TESTS:BOOL=FALSE"
)
