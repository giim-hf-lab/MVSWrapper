get_ext_version(nlohmann_json __L_VERSION)

add_ext_dep(nlohmann_json "${__G_GITHUB_PREFIX}" "nlohmann/json" "${__L_VERSION}" "v${__L_VERSION}" "" "CONFIG"
	"-D" "JSON_BuildTests:BOOL=FALSE"
	"-D" "JSON_GlobalUDLs:BOOL=FALSE"
	"-D" "JSON_ImplicitConversions:BOOL=FALSE"
	"-D" "JSON_SystemInclude:BOOL=TRUE"
)
