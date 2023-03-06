prefixed_option(BASLER_PYLON_VERSION "7" "Basler Pylon software version."
	TYPE STRING
)

get_prefixed_option(BASLER_PYLON_VERSION OUTPUT_PREFIXES "__L")

find_package(pylon "${__L_BASLER_PYLON_VERSION}" REQUIRED GLOBAL)

set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "pylon::pylon")
