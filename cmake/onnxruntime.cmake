include(common)

set(onnxruntime_BUILD_SHARED_LIB ON)
set(onnxruntime_USE_CUDA ON)
set(onnxruntime_CUDNN_HOME "/usr")
set(onnxruntime_USE_AVX ON)
set(onnxruntime_USE_AVX2 ON)
set(onnxruntime_USE_AVX512 ON)
set(onnxruntime_ENABLE_LTO ON)
set(onnxruntime_USE_NCCL ON)
set(onnxruntime_USE_MPI ON)

add_subdirectory(externals/onnxruntime/cmake EXCLUDE_FROM_ALL)
