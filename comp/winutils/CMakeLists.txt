if (NOT WIN32)
	return ()
endif ()

create_library_target(winutils "${__G_NAMESPACE}" __L_COMPONENT_NAME STATIC
	"source/strings.cpp"
)
target_include_directories("${__L_COMPONENT_NAME}"
	PUBLIC
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
)
