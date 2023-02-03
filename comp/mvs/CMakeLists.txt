# if (MSVC)
# 	create_msvc_cxx_module_target(mvs
# 		COMPILE_DEFINITIONS
# 			MSGPACK_USE_DEFINE_MAP
# 		SYSTEM_INCLUDE_DIRECTORIES
# 			${OPENCV_INCLUDE_DIRS}
# 		LINK_LIBRARIES
# 			opencv_core
# 			opencv_imgproc
# 
# 			mvs_api
# 	)
# 	add_library(mvs::mvs ALIAS mvs)
# else ()
# 	message(FATAL_ERROR "MVS wrapper requires C++20 module support")
# endif ()

# add_executable(mvs_driver EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
# target_link_libraries(mvs_driver
# 	PRIVATE
# 		mvs::mvs
# )
