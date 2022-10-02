cmake_minimum_required(VERSION 3.14)
cmake_policy(VERSION 3.14)
project(inferences LANGUAGES CXX C VERSION ${MASTER_PROJECT_VERSION})

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(PkgConfig REQUIRED)
if (onnxruntime_PREFIX)
	list(APPEND CMAKE_PREFIX_PATH ${onnxruntime_PREFIX})
endif ()
pkg_check_modules(onnxruntime REQUIRED libonnxruntime)

add_library(onnxruntime SHARED IMPORTED)
target_include_directories(onnxruntime INTERFACE ${onnxruntime_PREFIX}/include)
set_target_properties(onnxruntime PROPERTIES
	IMPORTED_LINK_INTERFACE_LANGUAGES CXX
	IMPORTED_LOCATION ${onnxruntime_LIBDIR}/libonnxruntime.so
)
target_link_directories(onnxruntime INTERFACE ${onnxruntime_LIBDIR})
target_link_libraries(onnxruntime INTERFACE ${onnxruntime_LINK_LIBRARIES})

if (IS_DIRECTORY ${CLIPPER_INSTALL_DIR})
	add_library(clipper STATIC IMPORTED)
	target_include_directories(clipper INTERFACE ${CLIPPER_INSTALL_DIR}/include)
	set_target_properties(clipper PROPERTIES
		IMPORTED_LINK_INTERFACE_LANGUAGES CXX
		IMPORTED_LOCATION ${CLIPPER_INSTALL_DIR}/lib/libClipper2.a
	)
	target_link_directories(clipper INTERFACE ${CLIPPER_INSTALL_DIR}/lib)
	target_link_libraries(clipper INTERFACE ${CLIPPER_INSTALL_DIR}/lib/libClipper2.a)
else ()
	message(FATAL_ERROR "CLIPPER_INSTALL_DIR is not valid: ${CLIPPER_INSTALL_DIR}.")
endif ()

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

find_package(OpenCV REQUIRED)

find_package(fmt REQUIRED)
find_package(mio REQUIRED)
find_package(nlohmann_json REQUIRED)

list(APPEND ${CMAKE_PROJECT_NAME}_TARGETS framework_onnxruntime)
set(framework_onnxruntime_LIBRARIES
	$<BUILD_INTERFACE:mio::mio>
	opencv_core
	opencv_imgproc

	clipper
	onnxruntime
)

list(APPEND ${CMAKE_PROJECT_NAME}_TARGETS framework_torchscript)
set(framework_torchscript_LIBRARIES
	$<BUILD_INTERFACE:fmt::fmt-header-only>
	$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
	opencv_core
	opencv_imgproc

	torch
	TorchVision::TorchVision
)

list(APPEND ${CMAKE_PROJECT_NAME}_TARGETS wrapper)

set(ALL_EXPORTED_TARGETS ${${CMAKE_PROJECT_NAME}_TARGETS})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(${CMAKE_PROJECT_NAME}_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(${CMAKE_PROJECT_NAME}_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/${CMAKE_PROJECT_NAME})

include(common)

foreach (target_name IN LISTS ${CMAKE_PROJECT_NAME}_TARGETS)
	string(REPLACE "_" "/" target_dir ${target_name})
	string(REPLACE "_" "::" target_export ${target_name})

	file(
		GLOB_RECURSE
			${target_name}_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/src/${target_dir}/*.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/${target_dir}/*.hpp
	)

	if (${CMAKE_PROJECT_NAME}_BUILD_SHARED)
		add_library(${target_name} SHARED ${${target_name}_SOURCES})
	else ()
		add_library(${target_name} STATIC ${${target_name}_SOURCES})
	endif ()
	set_target_properties(${target_name} PROPERTIES
		OUTPUT_NAME
			${CMAKE_PROJECT_NAME}_${target_name}
		EXPORT_NAME
			${target_export}
	)
	target_include_directories(${target_name} PRIVATE
		$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIR}>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	)
	target_link_libraries(${target_name} PRIVATE
		${${target_name}_LIBRARIES}
	)

	configure_package_config_file(
		cmake/config/${target_dir}.cmake.in
		${CMAKE_CURRENT_BINARY_DIR}/cmake/${CMAKE_PROJECT_NAME}/config/${target_dir}.cmake
		INSTALL_DESTINATION
			${CMAKE_INSTALL_DATADIR}/cmake/${CMAKE_PROJECT_NAME}/config
	)
endforeach ()

configure_package_config_file(
	cmake/${CMAKE_PROJECT_NAME}-config.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/cmake/${CMAKE_PROJECT_NAME}/${CMAKE_PROJECT_NAME}-config.cmake
	INSTALL_DESTINATION
		${CMAKE_INSTALL_DATADIR}/cmake/${CMAKE_PROJECT_NAME}
)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/cmake/${CMAKE_PROJECT_NAME}/${CMAKE_PROJECT_NAME}-config-version.cmake
	VERSION
		${${CMAKE_PROJECT_NAME}_VERSION}
	COMPATIBILITY
		AnyNewerVersion
)
install(
	DIRECTORY
		${CMAKE_CURRENT_BINARY_DIR}/cmake
	TYPE
		DATA
)

install(
	DIRECTORY
		${${CMAKE_PROJECT_NAME}_INCLUDE_DIR}/wrapper
	DESTINATION
		${${CMAKE_PROJECT_NAME}_INSTALL_INCLUDEDIR}
)
if (onnxruntime_FOUND)
	install(
		DIRECTORY
			${${CMAKE_PROJECT_NAME}_INCLUDE_DIR}/framework/onnxruntime
		DESTINATION
			${${CMAKE_PROJECT_NAME}_INSTALL_INCLUDEDIR}/framework
	)
endif ()
if (Torch_FOUND)
	install(
		DIRECTORY
			${${CMAKE_PROJECT_NAME}_INCLUDE_DIR}/framework/torchscript
		DESTINATION
			${${CMAKE_PROJECT_NAME}_INSTALL_INCLUDEDIR}/framework
	)
endif ()
install(
	TARGETS
		${${CMAKE_PROJECT_NAME}_TARGETS}
	EXPORT
		${CMAKE_PROJECT_NAME}-targets
)
install(
	EXPORT
		${CMAKE_PROJECT_NAME}-targets
	NAMESPACE
		inferences::
	DESTINATION
		${CMAKE_INSTALL_DATADIR}/cmake/${CMAKE_PROJECT_NAME}
)
