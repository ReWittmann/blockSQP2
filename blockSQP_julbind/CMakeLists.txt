project(Julia_BlockSQP)

cmake_minimum_required(VERSION 3.5)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

#Path to cxxwrap cmake file
list(APPEND CMAKE_PREFIX_PATH "/home/reinhold/.julia/artifacts/65c14d6c8b06e52ca794200129a8f3dd8b7ce34e/lib/cmake/JlCxx")
#list(APPEND CMAKE_PREFIX_PATH "/home/reinhold/.julia")

find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")

message(STATUS "Found JlCxx at ${JlCxx_location}")

add_library(BlockSQP SHARED BlockSQP.cpp)

target_include_directories(BlockSQP 
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(BlockSQP
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../lib/libblockSQP.so JlCxx::cxxwrap_julia)





