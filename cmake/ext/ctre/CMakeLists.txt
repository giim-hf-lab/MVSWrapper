get_ext_version(ctre __L_VERSION)

FetchContent_Declare(ctre
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}hanickadot/compile-time-regular-expressions.git"
	GIT_TAG "v${__L_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
	OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(ctre)

set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "ctre")
