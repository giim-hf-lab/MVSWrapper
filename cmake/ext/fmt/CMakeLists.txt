get_ext_version(fmt __L_VERSION)

add_ext_dep(fmt "${__G_GITHUB_PREFIX}" "fmtlib/fmt" "${__L_VERSION}" "${__L_VERSION}" "CXX" ""
	"-D" "FMT_DOC:BOOL=FALSE"
	"-D" "FMT_TEST:BOOL=FALSE"
	"-D" "FMT_MODULE:BOOL=TRUE"
	"-D" "FMT_SYSTEM_HEADERS:BOOL=TRUE"
)
