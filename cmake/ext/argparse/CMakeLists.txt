get_ext_version(argparse __L_VERSION)

FetchContent_Declare(argparse
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}p-ranav/argparse.git"
	GIT_TAG "v${__L_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
)

FetchContent_MakeAvailable(argparse)

set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "argparse")
