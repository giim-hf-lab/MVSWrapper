get_ext_version(zlib __L_VERSION)

set(ZLIB_USE_STATIC_LIBS TRUE)

add_ext_dep(ZLIB "${__G_GITHUB_PREFIX}" "madler/zlib" "${__L_VERSION}" "v${__L_VERSION}" "C" "")
