cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Please use out of source build:
  cmake -B build
  {make -C build } OR {cmake --build build}")
endif()

project(SUB_MUMPS)

include(GNUInstallDirs)

if (APPLE)
  set(rbase "@loader_path")
else ()
  set(rbase "$ORIGIN")
endif ()
file(RELATIVE_PATH lib_dir
     "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
     "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_INSTALL_RPATH "${rbase};${rbase}/${libdir}"
    CACHE STRING "Install RPATH")


include(FetchContent)
FetchContent_Declare(
	CMAKEMUMPS
	GIT_REPOSITORY https://github.com/scivision/mumps.git
	GIT_TAG b9ddf60306de7979ff42e5015b5eff70e28cdbed# refs/tags/v5.7.3.1
)
set(MUMPS_parallel no)
set(BUILD_SHARED_LIBS yes)
FetchContent_MakeAvailable(CMAKEMUMPS)

#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")

