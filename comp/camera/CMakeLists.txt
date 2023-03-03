create_library_target(camera "${__G_NAMESPACE}" __L_COMPONENT_NAME STATIC
	"source/basler.cpp"
	"source/mvs.cpp"
)
target_include_directories("${__L_COMPONENT_NAME}"
	PUBLIC
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries("${__L_COMPONENT_NAME}"
	PUBLIC
		opencv::core
		pylon::pylon
)
