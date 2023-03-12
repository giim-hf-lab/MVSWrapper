get_ext_version(sqlite3 __L_VERSION)

include("vars/${__L_VERSION}.cmake" NO_POLICY_SCOPE)

FetchContent_Declare(sqlite3
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}sqlite/sqlite.git"
	GIT_TAG "version-${__L_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
	OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(sqlite3)
FetchContent_GetProperties(sqlite3 SOURCE_DIR __L_GIT_DIR)

FetchContent_Declare(sqlite3-amalgamation
	URL "https://www.sqlite.org/${__L_FILE}"
	URL_HASH "SHA3_256=${__L_SHA3_256}"
	DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Populate(sqlite3-amalgamation)

FetchContent_GetProperties(sqlite3-amalgamation SOURCE_DIR __L_SOURCE_DIR BINARY_DIR __L_BINARY_DIR)

file(COPY
	"${__L_GIT_DIR}/ext/async/sqlite3async.h"
	"${__L_GIT_DIR}/ext/expert/sqlite3expert.h"
	"${__L_GIT_DIR}/ext/rbu/sqlite3rbu.h"
	"${__L_GIT_DIR}/ext/userauth/sqlite3userauth.h"
	DESTINATION "${__L_SOURCE_DIR}/include"
)
file(COPY
	"${__L_GIT_DIR}/ext/async/sqlite3async.c"
	"${__L_GIT_DIR}/ext/expert/sqlite3expert.c"
	"${__L_GIT_DIR}/ext/rbu/sqlite3rbu.c"
	DESTINATION "${__L_SOURCE_DIR}/src"
)
file(COPY
	"${__L_GIT_DIR}/ext/userauth/userauth.c"
	DESTINATION "${__L_SOURCE_DIR}"
)
configure_file("sqlite3.cmake" "${__L_SOURCE_DIR}/CMakeLists.txt" @ONLY)

add_subdirectory("${__L_SOURCE_DIR}" "${__L_BINARY_DIR}" EXCLUDE_FROM_ALL)

set_internal_cache(TARGETS_ENFORCE_SYSTEM TRUE "sqlite3")

get_archive_path("sqlite3" "${__L_BINARY_DIR}" __L_LIBRARY)

set(SQLite3_FOUND TRUE PARENT_SCOPE)
set(SQLite3_INCLUDE_DIRS "${__L_SOURCE_DIR}/include" PARENT_SCOPE)
set(SQLite3_LIBRARIES "${__L_LIBRARY}" PARENT_SCOPE)
set(SQLite3_VERSION "${__L_VERSION}" PARENT_SCOPE)
set(SQLite3_VERSION_STRING "${__L_VERSION}" PARENT_SCOPE)
