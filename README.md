# C++ Utilities Collection

This repository is intended to be used as a super project merged from two repositories.

All external dependencies are now included via the CMake `FetchContent` module available since version 3.11 instead of the old style project structure (Git submodules).

The original plan was to include C++20 module as part of the interface exposed, but the support of C++20 module is experimental in most of the toolchains (including compilers, IDEs and CMake itself). So the support of C++20 module has been dropped and only the legacy interface (C++ headers) will be provided.
