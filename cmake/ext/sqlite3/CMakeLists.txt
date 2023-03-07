get_ext_version(sqlite3 __L_VERSION)

include("vars/${__L_VERSION}.cmake" NO_POLICY_SCOPE)

FetchContent_Declare(sqlite3
	URL "https://www.sqlite.org/${__L_FILE}"
	URL_HASH "SHA3_256=${__L_SHA3_256}"
	DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Populate(sqlite3)

FetchContent_GetProperties(sqlite3 SOURCE_DIR __L_SOURCE_DIR BINARY_DIR __L_BINARY_DIR)

project(sqlite3
	VERSION "${__L_VERSION}"
	HOMEPAGE_URL "https://www.sqlite.org"
	LANGUAGES
		C
)

set(__L_INCLUDE_DIR "${__L_BINARY_DIR}/include")
file(COPY
	"${__L_SOURCE_DIR}/sqlite3.h"
	"${__L_SOURCE_DIR}/sqlite3ext.h"
	DESTINATION "${__L_INCLUDE_DIR}"
)

add_library(sqlite3 STATIC EXCLUDE_FROM_ALL "${__L_SOURCE_DIR}/sqlite3.c")
add_library(sqlite3::sqlite3 ALIAS sqlite3)

target_include_directories(sqlite3 PUBLIC "${__L_INCLUDE_DIR}")

set_property(
	TARGET sqlite3
	PROPERTY ARCHIVE_OUTPUT_DIRECTORY "${__L_BINARY_DIR}"
)
set_property(
	TARGET sqlite3
	PROPERTY COMPILE_PDB_OUTPUT_DIRECTORY "${__L_BINARY_DIR}"
)
set_property(
	TARGET sqlite3
	PROPERTY PDB_OUTPUT_DIRECTORY "${__L_BINARY_DIR}"
)

set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "sqlite3")
