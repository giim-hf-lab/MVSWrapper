get_ext_version(opencv __L_VERSION)

FetchContent_Declare(opencv_contrib
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}opencv/opencv_contrib.git"
	GIT_TAG "${__L_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
)
FetchContent_Populate(opencv_contrib)
FetchContent_GetProperties(opencv_contrib SOURCE_DIR __L_SOURCE_DIR BINARY_DIR __L_BINARY_DIR)

get_prefixed_option(OPENCV_CONTRIB_MODULES OUTPUT_PREFIXES "__L")
foreach (__L_MODULE ${__L_OPENCV_CONTRIB_MODULES})
	set(__L_EXTRA_MODULE_PATH "${__L_SOURCE_DIR}/modules/${__L_MODULE}")
	if (EXISTS "${__L_EXTRA_MODULE_PATH}")
		file(CREATE_LINK "${__L_EXTRA_MODULE_PATH}" "${__L_BINARY_DIR}/${__L_MODULE}" SYMBOLIC)
	else ()
		project_message(FATAL_ERROR "Extra module '${__L_MODULE}' does not exist.")
	endif ()
endforeach ()
