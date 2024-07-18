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
#Currently available: qpOASES (-DQPSOLVER_QPOASES), gurobi (-DQPSOLVER_GUROBI)
#IMPORTANT: They need to be the same as the ones used to compile blockSQP

SET(QPSOLVER_COMPILE_FLAGS -DQPSOLVER_QPOASES)# -DQPSOLVER_GUROBI)

target_include_directories(BlockSQP 
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include ${QPSOLVER_INCLUDE_DIRECTORIES})
target_compile_options(BlockSQP PUBLIC ${QPSOLVER_COMPILE_FLAGS})

target_link_libraries(BlockSQP
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../lib/libblockSQP.so)

