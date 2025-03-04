cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Please use out of source build:
  cmake -B build
  {make -C build } OR {cmake --build build}")
endif()

project(BlockSQP    
	DESCRIPTION "Sequential quadratic programming for problems with block-diagonal Hessian matrix."
	HOMEPAGE_URL https://github.com/djanka2/blockSQP
    	LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
	CMAKEMUMPS
	GIT_REPOSITORY https://github.com/scivision/mumps.git
	GIT_TAG b9ddf60306de7979ff42e5015b5eff70e28cdbed# refs/tags/v5.7.3.1
)
set(MUMPS_parallel no)
FetchContent_MakeAvailable(CMAKEMUMPS)


FetchContent_Declare(
	qpos
	GIT_REPOSITORY	https://github.com/coin-or/qpOASES.git
	GIT_TAG		680f18c8ef0018a120e1604b769f056e8368df97
)
FetchContent_GetProperties(qpos)
if(NOT qpos_POPULATED)
    FetchContent_Populate(qpos)
endif()

add_library(qpOASES STATIC
	${qpos_SOURCE_DIR}/src/Matrices.cpp          
	#${qpos_SOURCE_DIR}/src/SparseSolver.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/SparseSolver.cpp
	${qpos_SOURCE_DIR}/src/Bounds.cpp             
	${qpos_SOURCE_DIR}/src/MessageHandling.cpp   
	${qpos_SOURCE_DIR}/src/SQProblem.cpp
	${qpos_SOURCE_DIR}/src/Constraints.cpp        
	${qpos_SOURCE_DIR}/src/Options.cpp           
	${qpos_SOURCE_DIR}/src/SQProblemSchur.cpp
	${qpos_SOURCE_DIR}/src/Flipper.cpp            
	${qpos_SOURCE_DIR}/src/OQPinterface.cpp      
	${qpos_SOURCE_DIR}/src/SubjectTo.cpp
	${qpos_SOURCE_DIR}/src/Indexlist.cpp          
	${qpos_SOURCE_DIR}/src/QProblemB.cpp         
	${qpos_SOURCE_DIR}/src/Utils.cpp
	${qpos_SOURCE_DIR}/src/QProblem.cpp           
	${qpos_SOURCE_DIR}/src/SolutionAnalysis.cpp
)
target_compile_definitions(qpOASES PRIVATE SOLVER_MUMPS MUMPS_SEQ)
target_link_libraries(qpOASES PRIVATE MUMPS::MUMPS)

target_include_directories(qpOASES
	PRIVATE ${cmakemumps_SOURCE_DIR}/mumps/5.7.3/include ${cmakemumps_SOURCE_DIR}/mumps/5.7.3/libseq
	PUBLIC	${qpos_SOURCE_DIR}/include
	#INTERFACE ${qpos_SOURCE_DIR}/qpOASES/include ${qpos_SOURCE_DIR}/qpOASES/include/qpOASES
)
target_link_libraries(qpOASES PUBLIC ${MUMPS_BIN} dl)
if (UNIX)
	target_compile_options(qpOASES PUBLIC -fPIC)
endif()


add_library(blockSQP SHARED
	blockSQP/src/blocksqp_condensing.cpp
    	blockSQP/src/blocksqp_matrix.cpp
    	blockSQP/src/blocksqp_problemspec.cpp
    	blockSQP/src/blocksqp_general_purpose.cpp
    	blockSQP/src/blocksqp_glob.cpp
   	blockSQP/src/blocksqp_hess.cpp
   	blockSQP/src/blocksqp_iter.cpp
   	blockSQP/src/blocksqp_main.cpp
   	blockSQP/src/blocksqp_options.cpp
   	blockSQP/src/blocksqp_qp.cpp
   	blockSQP/src/blocksqp_restoration.cpp
   	blockSQP/src/blocksqp_stats.cpp
   	blockSQP/src/blocksqp_qpsolver.cpp
   	blockSQP/src/blocksqp_defs.cpp
   	)
target_compile_definitions(blockSQP PUBLIC QPSOLVER_QPOASES)
#target_compile_options(blockSQP PUBLIC -fPIC)
set_target_properties(blockSQP PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP)

target_include_directories(blockSQP PUBLIC blockSQP/include ${qpos_SOURCE_DIR}/include ${qpos_SOURCE_DIR}/include/qpOASES)
target_link_libraries(blockSQP PUBLIC qpOASES)

add_executable(example1 blockSQP/examples/example1.cpp)
target_compile_options(example1 PRIVATE -g -O0)
target_link_libraries(example1 blockSQP)
set_target_properties(example1
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP/examples")

add_executable(example_condensing blockSQP/examples/example_condensing.cpp)
target_compile_options(example_condensing PRIVATE -g -O0)
target_link_libraries(example_condensing blockSQP)
set_target_properties(example_condensing
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP/examples"
)

option(PYTHON_INTERFACE "build python interface" OFF)
set(PYTHON_PATH "" CACHE STRING "Path to Python install folder (default: empty)")

if(PYTHON_INTERFACE)
	if (NOT PYTHON_PATH STREQUAL "")
		message(STATUS "Got python path ${PYTHON_PATH}")
		set (PYTHON_PREFIX ${PYTHON_PATH})
		set (PYTHON_EXECUTABLE ${PYTHON_PREFIX}/bin/python3.13)
		set (PYTHON_LIBRARIES ${PYTHON_PREFIX}/lib/libpython3.13.a)
		set (PYTHON_INCLUDE_DIRS ${PYTHON_PREFIX}/include/python3.13/cpython)
		set (PYTHON_SITE_PACKAGES ${PYTHON_PREFIX}/python3.13/site-packages)
		set (PYTHON_MODULE_EXTENSION .so)
	endif()
	
	FetchContent_Declare(
		PYBIND11
		GIT_REPOSITORY https://github.com/pybind/pybind11.git
		GIT_TAG dd9ef9f901f3d49426f87a67666393970a6c9431
	)
	FetchContent_MakeAvailable(PYBIND11)
	
	pybind11_add_module(blockSQP_pybind python_Interface/blockSQP_pybind.cpp)
	target_include_directories(blockSQP_pybind PUBLIC blockSQP/include)
	target_link_libraries(blockSQP_pybind PUBLIC blockSQP)
	set_target_properties(blockSQP_pybind PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python_Interface)
endif()

option(JULIA_INTERFACE "build julia interface" OFF)
set(CXXWRAP_PATH "" CACHE STRING "Path to CxxWrap artifact libcxxwrap-julia (default: empty)")
#set(CXXWRAP_PATH /home/reinhold/.julia/artifacts/65c14d6c8b06e52ca794200129a8f3dd8b7ce34e/lib/cmake/JlCxx)
if(JULIA_INTERFACE)
	if(NOT CXXWRAP_PATH STREQUAL "")
		list(APPEND CMAKE_PREFIX_PATH ${CXXWRAP_PATH})
	endif()
	message(STATUS "cxxwrap path is ${CXXWRAP_PATH}")
	find_package(JlCxx)
	get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
	get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)

	message(STATUS "Found JlCxx at ${JlCxx_location}")
	add_library(blockSQP_jl SHARED julia_Interface/blockSQP_julbind.cpp)

	target_include_directories(blockSQP_jl PUBLIC blockSQP/include)
	target_link_libraries(blockSQP_jl PUBLIC blockSQP JlCxx::cxxwrap_julia)
	set_target_properties(blockSQP_jl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/julia_Interface)
endif()

