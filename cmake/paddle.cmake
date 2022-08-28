set(PADDLE_INFERENCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/paddle/paddle_inference_install_dir/paddle)

add_library(paddle_inference_static STATIC IMPORTED)
target_include_directories(paddle_inference_static INTERFACE
	${PADDLE_INFERENCE_DIR}/include
)
set_target_properties(paddle_inference_static PROPERTIES
	IMPORTED_LOCATION ${PADDLE_INFERENCE_DIR}/lib/libpaddle_inference.a
)
add_library(paddle::inference::static ALIAS
	paddle_inference_static
)

add_library(paddle_inference_shared SHARED IMPORTED)
target_include_directories(paddle_inference_shared INTERFACE
	${PADDLE_INFERENCE_DIR}/include
)
set_target_properties(paddle_inference_shared PROPERTIES
	IMPORTED_LOCATION ${PADDLE_INFERENCE_DIR}/lib/libpaddle_inference.so
)
add_library(paddle::inference::shared ALIAS
	paddle_inference_shared
)
