find_package(fmt CONFIG REQUIRED CONFIG GLOBAL)

find_package(OpenCV REQUIRED COMPONENTS core dnn highgui imgcodecs imgproc CONFIG GLOBAL)

find_package(HikVisionMVS MODULE REQUIRED GLOBAL BYPASS_PROVIDER)
find_package(HuarayMV MODULE REQUIRED GLOBAL BYPASS_PROVIDER)
find_package(pylon 7 CONFIG REQUIRED GLOBAL BYPASS_PROVIDER)

add_library("${CURRENT_PROJECT_NAME}_camera" STATIC
	"source/camera/basler.cpp"
	"source/camera/fake.cpp"
	"source/camera/hikvision.cpp"
	"source/camera/huaray.cpp"
)
add_library("${CURRENT_PROJECT_NAME}::camera" ALIAS "${CURRENT_PROJECT_NAME}_camera")

target_compile_definitions("${CURRENT_PROJECT_NAME}_camera"
	PRIVATE
		_UTILITIES_USE_FMT
)
target_include_directories("${CURRENT_PROJECT_NAME}_camera"
	PUBLIC
		"include"
)
target_link_libraries("${CURRENT_PROJECT_NAME}_camera"
	PRIVATE
		"${JBIG_LIBRARY}"
		"${LERC_LIBRARY}"
		fmt::fmt
		opencv_core
		opencv_highgui
		opencv_imgcodecs
		opencv_imgproc

		hikvision::mvs
		huaray::mv
		pylon::pylon
)
