get_ext_version(ctre __L_VERSION)

string(REPLACE "." ";" __L_VERSION_LIST "${__L_VERSION}")
list(GET __L_VERSION_LIST 0 __L_VERSION_MAJOR)

add_ext_dep(ctre "${__G_GITHUB_PREFIX}" "hanickadot/compile-time-regular-expressions" "${__L_VERSION_MAJOR}.0" "v${__L_VERSION}" "" "CONFIG"
	"-D" "CTRE_BUILD_TESTS:BOOL=FALSE"
	"-D" "CTRE_BUILD_PACKAGE:BOOL=FALSE"
	"-D" "CTRE_BUILD_PACKAGE_DEB:BOOL=FALSE"
	"-D" "CTRE_BUILD_PACKAGE_RPM:BOOL=FALSE"
)
