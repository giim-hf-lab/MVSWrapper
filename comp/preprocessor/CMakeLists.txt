create_library_target(preprocessor "${__G_NAMESPACE}" __L_COMPONENT_NAME INTERFACE)
target_include_directories("${__L_COMPONENT_NAME}"
	INTERFACE
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
)
