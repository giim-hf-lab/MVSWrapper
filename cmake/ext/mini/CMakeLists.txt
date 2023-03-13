get_ext_version(mini __L_VERSION)

FetchContent_Declare(mini
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}pulzed/mINI.git"
	GIT_TAG "${__L_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
)
FetchContent_Populate(mini)

FetchContent_GetProperties(mini SOURCE_DIR __L_SOURCE_DIR)

add_library(mini INTERFACE)
add_library(mini::mini ALIAS mini)

target_include_directories(mini INTERFACE "${__L_SOURCE_DIR}/src")

set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "mini")
