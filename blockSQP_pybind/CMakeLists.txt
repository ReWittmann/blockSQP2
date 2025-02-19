cmake_minimum_required(VERSION 3.18)
project(Python_BlockSQP)


#set (PYTHON_PREFIX .../PYTHONFOLDER) #path to python installation folder
#set (PYTHON_VERSION pythonx.xx) #e.g. python3.13
#set (PYTHON_MODULE_EXTENSION .a) #Static or shared

#set (PYTHON_EXECUTABLE ${PYTHON_PREFIX}/bin/${PYTHON_VERSION})
#set (PYTHON_LIBRARIES ${PYTHON_PREFIX}/lib/${PYTHON_VERSION}.a)
#set (Python_INCLUDE_DIRS ${PYTHON_PREFIX}/include/${PYTHON_VERSION}/cpython)
#set (PYTHON_SITE_PACKAGES ${PYTHON_PREFIX}/lib/python)

set (PYTHON_PREFIX /home/reinhold/.localpython)
set (PYTHON_EXECUTABLE /home/reinhold/.localpython/bin/python3.13)
set (PYTHON_LIBRARIES /home/reinhold/.localpython/lib/libpython3.13.a)
set (PYTHON_INCLUDE_DIRS /home/reinhold/.localpython/include/python3.13/cpython)
set (PYTHON_SITE_PACKAGES /home/reinhold/.localpython/lib/python3.13/site-packages)
set (PYTHON_MODULE_EXTENSION .a)
set(pybind11_DIR /home/reinhold/.localpython/lib/python3.13/site-packages/pybind11/share/cmake/pybind11)


find_package(pybind11 REQUIRED)


pybind11_add_module(BlockSQP 
	BlockSQP.cpp)

target_include_directories(BlockSQP 
	#PUBLIC ~/blockSQP_modified/include ${QPSOLVER_INCLUDE_DIRECTORIES})
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)# ${QPSOLVER_INCLUDE_DIRECTORIES})
	
#target_compile_options(BlockSQP PUBLIC ${QPSOLVER_COMPILE_FLAGS})

target_link_libraries(BlockSQP
	#PUBLIC ~/blockSQP_modified/lib/libblockSQP.so)
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../lib/libblockSQP.so)

