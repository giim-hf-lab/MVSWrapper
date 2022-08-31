set(PADDLE_INFERENCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/paddle/paddle_inference_install_dir/paddle)

add_library(paddle_inference SHARED IMPORTED)
target_include_directories(paddle_inference INTERFACE
	${PADDLE_INFERENCE_DIR}/include
)
set_target_properties(paddle_inference PROPERTIES
	IMPORTED_LOCATION ${PADDLE_INFERENCE_DIR}/lib/libpaddle_inference.so
)
add_library(paddle::inference ALIAS
	paddle_inference
)
