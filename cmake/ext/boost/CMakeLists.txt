set(__L_BOOST_REQUIRED_SUBMODULES "tools/cmake")
set(__L_BOOST_REQUIRED_COMPONENTS
	"align"
	"array"
	"assert"
	"config"
	"core"
	"io"
	"mpl"
	"predef"
	"preprocessor"
	"static_assert"
	"throw_exception"
	"type_traits"
	"utility"
)

list(REMOVE_DUPLICATES __L_BOOST_REQUIRED_COMPONENTS)
set(BOOST_INCLUDE_LIBRARIES ${__L_BOOST_REQUIRED_COMPONENTS})

set(BOOST_ENABLE_PYTHON OFF)
set(BOOST_RUNTIME_LINK "static")

set(BOOST_IOSTREAMS_ENABLE_ZLIB ON)

foreach (__L_COMPONENT ${__L_BOOST_REQUIRED_COMPONENTS})
	list(APPEND __L_BOOST_REQUIRED_SUBMODULES "libs/${__L_COMPONENT}")
endforeach ()

get_ext_version(boost __L_VERSION)

FetchContent_Declare(boost
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}boostorg/boost.git"
	GIT_TAG "boost-${__L_VERSION}"
	GIT_SUBMODULES
		${__L_BOOST_REQUIRED_SUBMODULES}
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
	OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(boost)

foreach (__L_MODULE ${__L_BOOST_REQUIRED_COMPONENTS})
	set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "boost_${__L_MODULE}")
endforeach ()
