set(USE_STANDARD_TMPFILE ON)

get_ext_version(xlsxwriter __L_VERSION)

FetchContent_Declare(xlsxwriter
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}jmcnamara/libxlsxwriter.git"
	GIT_TAG "RELEASE_${__L_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
)

FetchContent_MakeAvailable(xlsxwriter)

set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "xlsxwriter")

add_library(xlsxwriter::xlsxwriter ALIAS xlsxwriter)
