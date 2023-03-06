cmake_path(APPEND __G_GENERATED_FILES_DIR
	"opencv_extra_modules"
	OUTPUT_VARIABLE OPENCV_EXTRA_MODULES_PATH
)
file(MAKE_DIRECTORY "${OPENCV_EXTRA_MODULES_PATH}")

get_ext_version(opencv __L_VERSION)

FetchContent_Declare(opencv_contrib
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}opencv/opencv_contrib.git"
	GIT_TAG "${__L_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
)
FetchContent_Populate(opencv_contrib)

get_prefixed_option(OPENCV_CONTRIB_MODULES OUTPUT_PREFIXES "__L")

FetchContent_GetProperties(opencv_contrib SOURCE_DIR __L_OPENCV_CONTRIB_DIR)
foreach (__L_MODULE ${__L_OPENCV_CONTRIB_MODULES})
	cmake_path(APPEND __L_OPENCV_CONTRIB_DIR
		"modules"
		"${__L_MODULE}"
		OUTPUT_VARIABLE __L_EXTRA_MODULE_PATH
	)
	if (EXISTS "${__L_EXTRA_MODULE_PATH}")
		file(CREATE_LINK "${__L_EXTRA_MODULE_PATH}" "${OPENCV_EXTRA_MODULES_PATH}/${__L_MODULE}" SYMBOLIC)
	else ()
		project_message(FATAL_ERROR "Extra module '${__L_MODULE}' does not exist.")
	endif ()
endforeach ()
