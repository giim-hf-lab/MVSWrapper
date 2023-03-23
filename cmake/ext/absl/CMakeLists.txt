get_ext_version(absl __L_TAG)

string(REPLACE "." ";" __L_VERSION_LIST "${__L_TAG}")
list(GET __L_VERSION_LIST 0 __L_VERSION)

add_ext_dep(absl "${__G_GITHUB_PREFIX}" "abseil/abseil-cpp" "${__L_VERSION}" "${__L_TAG}" "" "CONFIG"
	"-D" "ABSL_PROPAGATE_CXX_STD:BOOL=FALSE"
	"-D" "ABSL_USE_SYSTEM_INCLUDES:BOOL=TRUE"
)
