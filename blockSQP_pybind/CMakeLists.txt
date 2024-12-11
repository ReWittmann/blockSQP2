cmake_minimum_required(VERSION 3.18)
project(Python_BlockSQP)


#set (Python_ROOT_DIR ...)
#set (Python_INCLUDE_DIR .../include/python3.10/cpython)

#set(DPYTHON_EXECUTABLE /path/to/python/.../bin/pythonx.xx)
#set(pybind11_DIR /path/to/pybind11/.../share/cmake/pybind11

find_package(pybind11 REQUIRED)


pybind11_add_module(BlockSQP 
	BlockSQP.cpp)

#SET(QPSOLVER_INCLUDE_DIRECTORIES /path/to/qpOASES/include /path/to/gurobi/.../include)
SET(QPSOLVER_INCLUDE_DIRECTORIES ~/qpOASES/include ~/gurobi/gurobi1102/linux64/include)
#IMPORTANT: Flags for qpsolvers (-DQPSOLVER_QPOASES -DQPSOLVER_GUROBI ...) have to be identical to those used
#to compile blockSQP
SET(QPSOLVER_COMPILE_FLAGS -DQPSOLVER_QPOASES -DQPSOLVER_GUROBI)

target_include_directories(BlockSQP 
	#PUBLIC ~/blockSQP_modified/include ${QPSOLVER_INCLUDE_DIRECTORIES})
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include ${QPSOLVER_INCLUDE_DIRECTORIES})
	
target_compile_options(BlockSQP PUBLIC ${QPSOLVER_COMPILE_FLAGS})

target_link_libraries(BlockSQP
	#PUBLIC ~/blockSQP_modified/lib/libblockSQP.so)
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../lib/libblockSQP.so)

