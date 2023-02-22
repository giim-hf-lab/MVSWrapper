prefixed_option(SECURE_MIMALLOC OFF "Use secure mimalloc.")
prefixed_option(SHARED_MIMALLOC "BUILD_SHARED_LIBS" "Build shared mimalloc."
	REFERENCE
)
prefixed_option(STATIC_MIMALLOC ON "Build static mimalloc.")
prefixed_option(TEST_MIMALLOC "BUILD_TESTING" "Build mimalloc test targets."
	REFERENCE
)

prefixed_option(USE_MIMALLOC ON "Use mimalloc as the default allocator.")

get_prefixed_option(SECURE_MIMALLOC OUTPUT_VARIABLE MI_SECURE)
get_prefixed_option(SHARED_MIMALLOC OUTPUT_VARIABLE MI_BUILD_SHARED)
get_prefixed_option(STATIC_MIMALLOC OUTPUT_VARIABLE MI_BUILD_STATIC)
set(MI_BUILD_OBJECT ON)
get_prefixed_option(TEST_MIMALLOC OUTPUT_VARIABLE MI_BUILD_TESTS)
set(MI_USE_CXX ON)

FetchContent_Declare(mimalloc
	GIT_REPOSITORY "$CACHE{__C_GITHUB_PREFIX}microsoft/mimalloc.git"
	GIT_TAG "v${MIMALLOC_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
	OVERRIDE_FIND_PACKAGE
)

find_package(mimalloc REQUIRED GLOBAL)

set(__L_TARGETS "obj")
if (MI_BUILD_STATIC)
	list(APPEND __L_TARGETS "static")
endif ()
if (MI_BUILD_SHARED)
	list(APPEND __L_TARGETS "shared")
endif ()

foreach (__L_TARGET ${__L_TARGETS})
	set_property(
		TARGET
			"mimalloc-${__L_TARGET}"
		PROPERTY SYSTEM TRUE
	)
	add_library("mimalloc::${__L_TARGET}" ALIAS "mimalloc-${__L_TARGET}")
endforeach ()
