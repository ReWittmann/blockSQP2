cmake_minimum_required(VERSION 3.18)
project(Python_BlockSQP)

find_package(pybind11 REQUIRED)


pybind11_add_module(BlockSQP 
	BlockSQP.cpp)



target_include_directories(BlockSQP 
	PUBLIC ~/blockSQP_modified/include)


target_link_libraries(BlockSQP
	PUBLIC ~/blockSQP_modified/lib/libblockSQP.so)

