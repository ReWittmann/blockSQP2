project(Julia_BlockSQP)

cmake_minimum_required(VERSION 3.5)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
SET(QPSOLVER_INCLUDE_DIRECTORIES ~/qpOASES/include ~/gurobi/gurobi1102/linux64/include)


#Path to cxxwrap cmake file
list(APPEND CMAKE_PREFIX_PATH "/home/reinhold/.julia/artifacts/65c14d6c8b06e52ca794200129a8f3dd8b7ce34e/lib/cmake/JlCxx")


find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")

message(STATUS "Found JlCxx at ${JlCxx_location}")

add_library(BlockSQP SHARED BlockSQP.cpp)

#IMPORTANT: Flags for qpsolvers (-DQPSOLVER_QPOASES -DQPSOLVER_GUROBI ...) have to be identical to those used
#to compile blockSQP, else julia will crash when trying to create SQPoptions object
target_compile_options(BlockSQP PUBLIC -DQPSOLVER_QPOASES -DQPSOLVER_GUROBI)


target_include_directories(BlockSQP 
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include ${QPSOLVER_INCLUDE_DIRECTORIES})
target_link_libraries(BlockSQP
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../lib/libblockSQP.so JlCxx::cxxwrap_julia)



#target_link_libraries(BlockSQP JlCxx::cxxwrap_julia)



