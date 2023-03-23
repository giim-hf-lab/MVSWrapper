get_ext_version(sqlite3 __L_VERSION)

include("vars/${__L_VERSION}.cmake" NO_POLICY_SCOPE)

FetchContent_Declare(sqlite3
	GIT_REPOSITORY "${__G_GITHUB_PREFIX}sqlite/sqlite.git"
	GIT_TAG "version-${__L_VERSION}"
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW "${__G_FETCHCONTENT_GIT_SHALLOW}"
	GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
)
FetchContent_Populate(sqlite3)

FetchContent_GetProperties(sqlite3 SOURCE_DIR __L_GIT_DIR)

FetchContent_Declare(sqlite3-amalgamation
	URL "https://www.sqlite.org/${__L_FILE}"
	URL_HASH "SHA3_256=${__L_SHA3_256}"
	DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Populate(sqlite3-amalgamation)

FetchContent_GetProperties(sqlite3-amalgamation SOURCE_DIR __L_SOURCE_DIR BINARY_DIR __L_BINARY_DIR)

file(WRITE "${__L_SOURCE_DIR}/include/sqlite3.h"
[[
#ifndef SQLITE3_FEATURE_DEFINITIONS
#define SQLITE3_FEATURE_DEFINITIONS

#define SQLITE_ENABLE_ASYNCIO 1
#define SQLITE_ENABLE_ATOMIC_WRITE 1
#define SQLITE_ENABLE_BATCH_ATOMIC_WRITE 1
#define SQLITE_ENABLE_FTS3_PARENTHESIS 1
#define SQLITE_ENABLE_FTS4 1
#define SQLITE_ENABLE_FTS5 1
#define SQLITE_ENABLE_GEOPOLY 1
#define SQLITE_ENABLE_HIDDEN_COLUMNS 1
#define SQLITE_ENABLE_MATH_FUNCTIONS 1
#define SQLITE_ENABLE_MEMORY_MANAGEMENT 1
#define SQLITE_ENABLE_MEMSYS5 1
#define SQLITE_ENABLE_NORMALIZE 1
#define SQLITE_ENABLE_PREUPDATE_HOOK 1
#define SQLITE_ENABLE_RBU 1
#define SQLITE_ENABLE_RTREE 1
#define SQLITE_ENABLE_SESSION 1
#define SQLITE_ENABLE_SNAPSHOT 1
#define SQLITE_ENABLE_UNLOCK_NOTIFY 1
#define SQLITE_USE_ALLOCA 1
#define SQLITE_USER_AUTHENTICATION 1

#ifdef _WIN32
#define SQLITE_WIN32_MALLOC 1
#endif

#endif

]]
)
file(READ "${__L_SOURCE_DIR}/sqlite3.h" __L_SQLITE3_H)
file(APPEND "${__L_SOURCE_DIR}/include/sqlite3.h" "${__L_SQLITE3_H}")
file(COPY
	"${__L_GIT_DIR}/ext/async/sqlite3async.h"
	"${__L_GIT_DIR}/ext/expert/sqlite3expert.h"
	"${__L_GIT_DIR}/ext/rbu/sqlite3rbu.h"
	"${__L_GIT_DIR}/ext/userauth/sqlite3userauth.h"
	DESTINATION "${__L_SOURCE_DIR}/include"
)
file(COPY
	"${__L_SOURCE_DIR}/sqlite3.c"
	"${__L_GIT_DIR}/ext/async/sqlite3async.c"
	"${__L_GIT_DIR}/ext/expert/sqlite3expert.c"
	"${__L_GIT_DIR}/ext/rbu/sqlite3rbu.c"
	DESTINATION "${__L_SOURCE_DIR}/src"
)
file(READ "${__L_GIT_DIR}/ext/userauth/userauth.c" __L_USERAUTH_C)
file(APPEND "${__L_SOURCE_DIR}/src/sqlite3.c" "${__L_USERAUTH_C}")
configure_file("sqlite3.cmake" "${__L_SOURCE_DIR}/CMakeLists.txt" @ONLY)

ext_compile_install(sqlite3 "${__L_SOURCE_DIR}" "" "${__L_BINARY_DIR}")

find_package(SQLite3 "${__L_VERSION}" REQUIRED GLOBAL BYPASS_PROVIDER)
