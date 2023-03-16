prefixed_option(BASLER_PYLON_VERSION "7" "Basler Pylon software version."
	TYPE STRING
)

get_prefixed_option(BASLER_PYLON_VERSION OUTPUT_PREFIXES "__L")

find_package(pylon "${__L_BASLER_PYLON_VERSION}" REQUIRED GLOBAL)

if (MSVC)
	target_link_options(pylon::pylon
		INTERFACE
			"/NODEFAULTLIB:GCBase_MDd_VC141_v3_1_Basler_pylon.lib"
			"/NODEFAULTLIB:GenApi_MDd_VC141_v3_1_Basler_pylon.lib"
	)
endif ()
