cmake_minimum_required(VERSION 3.18)
project(Python_BlockSQP)


#set (Python_ROOT_DIR ...)
#set (Python_INCLUDE_DIR .../include/python3.10/cpython)

#set(DPYTHON_EXECUTABLE /path/to/python/.../bin/pythonx.xx)
#set(pybind11_DIR /path/to/pybind11/.../share/cmake/pybind11

find_package(pybind11 REQUIRED)


pybind11_add_module(BlockSQP 
	BlockSQP.cpp)

#Edit here and add paths to qp solver include directories
#SET(QPSOLVER_INCLUDE_DIRECTORIES /path/to/.../qpOASES/include /path/to/gurobi/.../include ...)

#Add flags for inclusion of qp solvers
#IMPORTANT: They need to be the same as the ones used to compile blockSQP to libblockSQP.so
SET(QPSOLVER_COMPILE_FLAGS -DQPSOLVER_QPOASES)# -DQPSOLVER_GUROBI)

target_include_directories(BlockSQP 
	PUBLIC ~/blockSQP_modified/include ${QPSOLVER_INCLUDE_DIRECTORIES})
target_compile_options(BlockSQP PUBLIC ${QPSOLVER_COMPILE_FLAGS})

target_link_libraries(BlockSQP
	PUBLIC ~/blockSQP_modified/lib/libblockSQP.so)

