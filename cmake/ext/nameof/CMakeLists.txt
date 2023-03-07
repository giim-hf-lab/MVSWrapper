get_ext_version(nameof __L_VERSION)

FetchContent_Declare(nameof
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}Neargye/nameof.git"
	GIT_TAG "v${__L_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
)

FetchContent_MakeAvailable(nameof)

set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "nameof")
