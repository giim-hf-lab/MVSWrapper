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
		mvs::mvs
		opencv::core
		opencv::imgcodecs
		pylon::pylon
)

create_example(camera "${__G_NAMESPACE}" __L_COMPONENT_EXAMPLE)
target_link_libraries("${__L_COMPONENT_EXAMPLE}"
	PRIVATE
		${__L_COMPONENT_NAME}
)
