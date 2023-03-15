create_library_target(camera "${__G_NAMESPACE}" __L_COMPONENT_NAME STATIC
	"source/camera/basler.cpp"
	"source/camera/fake.cpp"
	"source/camera/mvs.cpp"
)
target_include_directories("${__L_COMPONENT_NAME}"
	PUBLIC
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries("${__L_COMPONENT_NAME}"
	PUBLIC
		mvs::mvs
		opencv::core
		opencv::imgcodecs
		pylon::pylon
)
