prefixed_option(MI_BUILD_SHARED "BUILD_SHARED_LIBS" "Build shared mimalloc."
	REFERENCE
)
prefixed_option(MI_BUILD_STATIC ON "Build static mimalloc.")
prefixed_option(MI_BUILD_TESTS "BUILD_TESTING" "Build mimalloc test targets."
	REFERENCE
)

set(MI_OVERRIDE ON)
get_prefixed_option(MI_BUILD_SHARED)
get_prefixed_option(MI_BUILD_STATIC)
set(MI_BUILD_OBJECT ON)
get_prefixed_option(MI_BUILD_TESTS)
set(MI_USE_CXX ON)

get_ext_version(mimalloc __L_VERSION)

FetchContent_Declare(mimalloc
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}microsoft/mimalloc.git"
	GIT_TAG "v${__L_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
)

FetchContent_MakeAvailable(mimalloc)

set(__L_TARGETS "obj")
if (MI_BUILD_STATIC)
	list(APPEND __L_TARGETS "static")
endif ()
if (MI_BUILD_SHARED)
	list(APPEND __L_TARGETS "shared")
endif ()

foreach (__L_TARGET ${__L_TARGETS})
	set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "mimalloc-${__L_TARGET}")
	add_library("mimalloc::${__L_TARGET}" ALIAS "mimalloc-${__L_TARGET}")
endforeach ()
