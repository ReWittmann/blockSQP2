cmake_minimum_required(VERSION 3.18)
project(Python_BlockSQP)

find_package(pybind11 REQUIRED)


pybind11_add_module(BlockSQP 
	BlockSQP.cpp)


########################################
#Paths to blockSQP need to be added here
########################################
target_include_directories(BlockSQP 
	PUBLIC PATH/TO/BLOCKSQP/include)


target_link_libraries(BlockSQP
	PUBLIC PATH/TO/BLOCKSQP/lib/libblockSQP.so)

