include_guard(GLOBAL)

prepare("ext" "" "zlib" "png")

prefixed_option(OPENCV_CONTRIB_MODULES "" "OpenCV Contrib modules to be included as part of OpenCV."
	TYPE STRING
)

get_prefixed_option(OPENCV_CONTRIB_MODULES OUTPUT_PREFIXES "__L")

set(__L_MODULES)
if (__L_OPENCV_CONTRIB_MODULES)
	prepare("ext" "opencv_contrib/" ${__L_OPENCV_CONTRIB_MODULES})
	list(APPEND __L_MODULES "opencv_contrib")
endif ()
list(APPEND __L_MODULES "opencv")

set_internal_cache(EXPORTED_EXTERNAL_DEPENDENCIES TRUE ${__L_MODULES})
