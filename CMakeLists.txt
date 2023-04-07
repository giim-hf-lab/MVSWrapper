cmake_minimum_required(VERSION 3.25)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CURRENT_PROJECT_NAME "utilities")

project("${CURRENT_PROJECT_NAME}"
	VERSION 0.0
	LANGUAGES
		CXX
)

find_program(VCPKG_EXECUTABLE "vcpkg" PATHS "$ENV{VCPKG_ROOT}" REQUIRED NO_DEFAULT_PATH)

execute_process(
	COMMAND
		"${VCPKG_EXECUTABLE}"
		"install"
		"--recurse"
		"--triplet" "${VCPKG_HOST_TRIPLET}-static"
		"abseil[cxx17]"
		"antlr4"
		"boost[mpi]"
		"boost-locale[icu]"
		"ceres[cuda,cxsparse,eigensparse,lapack,suitesparse]"
		"curl[brotli,c-ares,http2,idn,mbedtls,non-http,openssl,schannel,ssh,ssl,sspi,tool,websockets,winidn,winldap,winssl]"
		"fmt"
		"folly[bzip2,libsodium,lz4,lzma,snappy,zlib,zstd]"
		"freetype[brotli,bzip2,error-strings,png,zlib]"
		"gperftools[override]"
		"hdf5[cpp,map,szip,zlib]"
		"jemalloc"
		"jsoncpp"
		"leveldb[crc32c,snappy]"
		"libarchive[bzip2,crypto,libxml2,lz4,lzma,lzo,openssl,zstd]"
		"libpq[icu,lz4,openssl,xml,xslt,zlib,zstd]"
		"libevent[openssl,thread]"
		"libjpeg-turbo[jpeg7,jpeg8]"
		"libsrt[bonding]"
		"libssh[crypto,mbedtls,openssl,zlib]"
		"libssh2[zlib]"
		"libtomcrypt"
		"libxlsxwriter"
		"libvpx[highbitdepth,realtime]"
		"libxml2[ftp,http,iconv,lzma,zlib]"
		"libyaml"
		"libzip[bzip2,commoncrypto,default-aes,liblzma,mbedtls,openssl,wincrypto,zstd]"
		"mbedtls[pthreads]"
		"mimalloc[override]"
		"msgpack"
		"nlohmann-json"
		"openblas[simplethread,threads]"
		"opencv[ade,cuda,cudnn,dnn,dnn-cuda,eigen,freetype,jpeg,lapack,nonfree,openmp,png,quirc,sfm,tiff,webp]"
		"opus[avx]"
		"pkgconf"
		"protobuf[zlib]"
		"rapidjson"
		"ryml"
		"sdl2[base,samplerate]"
		"soxr[lsr-bindings,openmp]"
		"spdlog[wchar]"
		"sqlite3[fts3,fts4,fts5,geopoly,json1,limit,math,memsys5,rtree,session,tool,zlib]"
		"tiff[cxx,jpeg,lzma,zip,zstd]"
		"x264[asm,chroma-format-all,default-features]"
		"xlnt"
		"yaml-cpp"
	OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_install.${VCPKG_HOST_TRIPLET}-static.log"
	ERROR_FILE "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_install.${VCPKG_HOST_TRIPLET}-static.log"
	ENCODING "UTF-8"
	COMMAND_ERROR_IS_FATAL "ANY"
)

execute_process(
	COMMAND
		"${VCPKG_EXECUTABLE}"
		"install"
		"--recurse"
		"--triplet" "${VCPKG_HOST_TRIPLET}"
		"onnxruntime-gpu"
	OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_install.${VCPKG_HOST_TRIPLET}.log"
	ERROR_FILE "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_install.${VCPKG_HOST_TRIPLET}.log"
	ENCODING "UTF-8"
	COMMAND_ERROR_IS_FATAL "ANY"
)

add_subdirectory("comp/camera" EXCLUDE_FROM_ALL)
