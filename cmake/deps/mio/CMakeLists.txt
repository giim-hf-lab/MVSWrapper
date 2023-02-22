FetchContent_Declare(mio
	GIT_REPOSITORY "$CACHE{__C_GITHUB_PREFIX}laitingsheng/mio.git"
	GIT_TAG "${MIO_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
	OVERRIDE_FIND_PACKAGE
)

find_package(mio REQUIRED GLOBAL)

set_property(
	TARGET
		mio
		mio-headers
	PROPERTY SYSTEM TRUE
)
if (WIN32)
	set_property(
		TARGET
			mio_full_winapi
			mio_min_winapi
		PROPERTY SYSTEM TRUE
	)
endif ()
