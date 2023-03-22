cmake_minimum_required(VERSION 3.1)

project(sqlite3
	VERSION "@__L_VERSION@"
	HOMEPAGE_URL "https://www.sqlite.org"
	LANGUAGES
		C
)

include(GNUInstallDirs)

add_library(sqlite3 STATIC
	"src/sqlite3.c"
	"src/sqlite3async.c"
	"src/sqlite3expert.c"
	# "src/sqlite3rbu.c"
)
set_property(
	TARGET
		sqlite3
	PROPERTY PUBLIC_HEADER
		"${CMAKE_INSTALL_INCLUDEDIR}/sqlite3.h"
		"${CMAKE_INSTALL_INCLUDEDIR}/sqlite3async.h"
		"${CMAKE_INSTALL_INCLUDEDIR}/sqlite3expert.h"
		"${CMAKE_INSTALL_INCLUDEDIR}/sqlite3rbu.h"
		"${CMAKE_INSTALL_INCLUDEDIR}/sqlite3userauth.h"
)
target_include_directories(sqlite3
	PUBLIC
		"${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
	TARGETS
		sqlite3
)
