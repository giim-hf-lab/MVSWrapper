set(JSON_GlobalUDLs OFF)
set(JSON_ImplicitConversions OFF)

get_ext_version(nlohmann_json __L_VERSION)

FetchContent_Declare(nlohmann_json
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}nlohmann/json.git"
	GIT_TAG "v${__L_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
)

FetchContent_MakeAvailable(nlohmann_json)

set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "nlohmann_json")
