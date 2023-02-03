string(REPLACE "." "-" __L_FREETYPE_VERSION_STRING "${FREETYPE_VERSION}")

set(FT_DISABLE_ZLIB ON)
set(FT_ENABLE_ERROR_STRINGS ON)

FetchContent_Declare(freetype
	GIT_REPOSITORY "$CACHE{__C_GITHUB_PREFIX}freetype/freetype.git"
	GIT_TAG "VER-${__L_FREETYPE_VERSION_STRING}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
	OVERRIDE_FIND_PACKAGE
)

find_package(freetype REQUIRED GLOBAL)

set_property(
	TARGET
		freetype
	PROPERTY SYSTEM TRUE
)
