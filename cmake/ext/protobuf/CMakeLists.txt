get_ext_version(protobuf __L_VERSION)

add_ext_dep(protobuf "${__G_GITHUB_PREFIX}" "protocolbuffers/protobuf" "" "v${__L_VERSION}" "" ""
	"-D" "ZLIB_USE_STATIC_LIBS:BOOL=TRUE"
	"-D" "protobuf_BUILD_TESTS:BOOL=FALSE"
	"-D" "protobuf_BUILD_LIBPROTOC:BOOL=TRUE"
	"-D" "protobuf_ABSL_PROVIDER:STRING=package"
)

set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf REQUIRED CONFIG GLOBAL BYPASS_PROVIDER)
