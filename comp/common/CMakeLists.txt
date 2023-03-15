set(__L_SOURCES
	"source/common/ini.cpp"
	"source/common/stopwatch.cpp"
)
if (WIN32)
	list(APPEND __L_SOURCES
		"source/common/std/win.cpp"
	)
endif ()

create_library_target(common "${__G_NAMESPACE}" __L_COMPONENT_NAME STATIC ${__L_SOURCES})
target_include_directories("${__L_COMPONENT_NAME}"
	PUBLIC
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries("${__L_COMPONENT_NAME}"
	PUBLIC
		mini::mini
)
