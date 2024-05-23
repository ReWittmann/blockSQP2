project(Julia_BlockSQP)

cmake_minimum_required(VERSION 3.5)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

#Add path to JlCxx library
list(APPEND CMAKE_PREFIX_PATH "PATH/TO/.julia/artifacts/5209ca23f516fb3391b885eef717e49b4ee0a268/lib/cmake/JlCxx")

find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")

message(STATUS "Found JlCxx at ${JlCxx_location}")

add_library(BlockSQP SHARED BlockSQP.cpp)
target_include_directories(BlockSQP 
	PUBLIC PATH/TO/BLOCKSQP/include)
target_link_libraries(BlockSQP
	PUBLIC PATH/TO/BLOCKSQP/lib/libblockSQP.so JlCxx::cxxwrap_julia)



#target_link_libraries(BlockSQP JlCxx::cxxwrap_julia)



