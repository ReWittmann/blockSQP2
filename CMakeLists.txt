cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Please use out of source build:
  cmake -B build
  {make -C build } OR {cmake --build build}")
endif()

project(BlockSQP    
	DESCRIPTION "Sequential quadratic programming for problems with block-diagonal Hessian matrix."
	HOMEPAGE_URL https://github.com/djanka2/blockSQP
    	LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 20)
enable_language(Fortran)

#set(N_LINSOL_BIN -1 CACHE STRING "The number of created MUMPS binaries for calling in different threads (MUMPS is not thread safe)")

#[[
include(ExternalProject)

foreach(i RANGE 0 7)
	ExternalProject_Add(
	SUB_MUMPS_${i}
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/MUMPS_SUBDIRS/${i}
	BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/MUMPS_SUBDIRS/${i}/build
	CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/MUMPS_SUBDIRS/${i}/install
	INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/MUMPS_SUBDIRS/${i}/install
	BUILD_ALWAYS 1
	)
	set(MUMPS_PATH_${i} ${CMAKE_CURRENT_SOURCE_DIR}/MUMPS_SUBDIRS/${i}/install/lib/libdmumps.so)
	message(STATUS MUMPS_PATH_${i} " " is " " ${MUMPS_PATH_${i}})
endforeach()
]]

#set(MUMPS_PATH_0 ${CMAKE_CURRENT_SOURCE_DIR}/MUMPS_SUBDIRS/0/install/lib/0_libdmumps.so.5.7.3.1)
#set(MUMPS_PATH_1 ${CMAKE_CURRENT_SOURCE_DIR}/MUMPS_SUBDIRS/1/install/lib/1_libdmumps.so.5.7.3.1)
#message(STATUS MUMPS_PATH_1 " " is " " ${MUMPS_PATH_1})


#set(BLA_VENDOR OpenBLAS)
#set(BLA_STATIC ON)

include(FetchContent)
FetchContent_Declare(
	CMAKEMUMPS
	GIT_REPOSITORY https://github.com/scivision/mumps.git
	GIT_TAG 45c16604c1d8b61134699db04b16aae7b4704fc5# refs/tags/v5.7.3.1
)
set(MUMPS_parallel OFF)
set(BUILD_SHARED_LIBS OFF)
set(MUMPS_scalapack OFF)
set(BUILD_SINGLE OFF)
set(MUMPS_find_static ON)

FetchContent_MakeAvailable(CMAKEMUMPS)


#[[
FetchContent_Declare(
	qpos
	GIT_REPOSITORY	https://github.com/coin-or/qpOASES.git
	GIT_TAG		680f18c8ef0018a120e1604b769f056e8368df97
	SOURCE_SUBDIR 	DIR/THAT/DOESNT/EXIST
)
FetchContent_GetProperties(qpos)
if(NOT qpos_POPULATED)
	FetchContent_MakeAvailable(qpos)
endif()

add_library(qpOASES STATIC
	${qpos_SOURCE_DIR}/src/Matrices.cpp          
	${qpos_SOURCE_DIR}/src/SparseSolver.cpp
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
	${qpos_SOURCE_DIR}/src/QProblemB.cpp         
	${qpos_SOURCE_DIR}/src/Utils.cpp
	${qpos_SOURCE_DIR}/src/QProblem.cpp     
	${qpos_SOURCE_DIR}/src/QProblem.cpp           
	${qpos_SOURCE_DIR}/src/SolutionAnalysis.cpp
)
target_compile_definitions(qpOASES PRIVATE SOLVER_MUMPS MUMPS_SEQ __NO_COPYRIGHT__)
target_compile_options(qpOASES PRIVATE -O3)
target_link_libraries(qpOASES PRIVATE MUMPS::MUMPS)
IF(UNIX AND NOT APPLE)
   target_compile_definitions(qpOASES PRIVATE LINUX)
ELSEIF(WINDOWS)
    target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
ENDIF()


target_include_directories(qpOASES
	PRIVATE ${cmakemumps_SOURCE_DIR}/mumps/5.7.3/include ${cmakemumps_SOURCE_DIR}/mumps/5.7.3/libseq
	PUBLIC	${qpos_SOURCE_DIR}/include
	#PUBLIC	${CMAKE_CURRENT_SOURCE_DIR}/modded_external
)
target_link_libraries(qpOASES PUBLIC ${MUMPS_BIN} dl)
if (UNIX)
	target_compile_options(qpOASES PUBLIC -fPIC)
endif()
]]



set (qpOASES_MODDED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/modded_external/qpOASES_mod)
add_library(qpOASES STATIC
	${qpOASES_MODDED_DIR}/src/Matrices.cpp          
	${qpOASES_MODDED_DIR}/src/SparseSolver.cpp
	${qpOASES_MODDED_DIR}/src/Bounds.cpp             
	${qpOASES_MODDED_DIR}/src/MessageHandling.cpp   
	${qpOASES_MODDED_DIR}/src/SQProblem.cpp
	${qpOASES_MODDED_DIR}/src/Constraints.cpp        
	${qpOASES_MODDED_DIR}/src/Options.cpp           
	${qpOASES_MODDED_DIR}/src/SQProblemSchur.cpp
	${qpOASES_MODDED_DIR}/src/Flipper.cpp            
	${qpOASES_MODDED_DIR}/src/OQPinterface.cpp      
	${qpOASES_MODDED_DIR}/src/SubjectTo.cpp
	${qpOASES_MODDED_DIR}/src/Indexlist.cpp          
	${qpOASES_MODDED_DIR}/src/QProblemB.cpp           
	${qpOASES_MODDED_DIR}/src/Utils.cpp
	${qpOASES_MODDED_DIR}/src/QProblem.cpp            
	${qpOASES_MODDED_DIR}/src/SolutionAnalysis.cpp
	)


#MUMPS LINEAR SOLVER
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)

target_compile_definitions(qpOASES PUBLIC SOLVER_MUMPS PRIVATE MUMPS_SEQ __NO_COPYRIGHT__)
target_compile_options(qpOASES PRIVATE -O3 INTERFACE -Wno-overloaded-virtual)
target_link_libraries(qpOASES PUBLIC MUMPS::MUMPS)
#target_link_libraries(qpOASES PUBLIC LAPACK::LAPACK BLAS::BLAS)
IF(UNIX)
	target_compile_definitions(qpOASES PRIVATE LINUX)
	if (NOT APPLE)
		target_link_libraries(qpOASES PUBLIC -lgfortran)
	endif()
ELSEIF(WINDOWS)
    target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
ENDIF()
target_include_directories(qpOASES
	PUBLIC	${qpOASES_MODDED_DIR}/include
)

target_include_directories(qpOASES
	PUBLIC	${qpOASES_MODDED_DIR}/include ${qpOASES_MODDED_DIR}/include/qpOASES
	#PUBLIC	${CMAKE_CURRENT_SOURCE_DIR}/modded_external
)

if (UNIX)
	target_compile_options(qpOASES PUBLIC -fPIC)
endif()



#MA57 LINEAR SOLVER
#[[
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_compile_definitions(qpOASES PRIVATE SOLVER_MA57 __NO_COPYRIGHT__)
target_compile_options(qpOASES PRIVATE -O3)
IF(UNIX AND NOT APPLE)
   target_compile_definitions(qpOASES PRIVATE LINUX)
ELSEIF(WINDOWS)
    target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
ENDIF()

target_include_directories(qpOASES
	PUBLIC	${qpOASES_MODDED_DIR}/include
)
target_link_libraries(qpOASES PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libcoinhsl.so /home/reinhold/LIB/install/lib/libmetis.a /home/reinhold/LIB/install/lib/libGKlib.a LAPACK::LAPACK BLAS::BLAS dl)
]]


#SPRAL LINEAR SOLVER
#[[
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_compile_definitions(qpOASES PRIVATE SOLVER_SPRAL __NO_COPYRIGHT__)
target_compile_options(qpOASES PRIVATE -O3)
IF(UNIX AND NOT APPLE)
   target_compile_definitions(qpOASES PRIVATE LINUX)
ELSEIF(WINDOWS)
    target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
ENDIF()

target_include_directories(qpOASES
	PUBLIC	${qpOASES_MODDED_DIR}/include /path/to/.../spral/include
)
target_link_libraries(qpOASES PUBLIC /path/to/.../libspral.so dl)
]]





add_library(blockSQP SHARED
	blockSQP/src/blocksqp_condensing.cpp
	blockSQP/src/blocksqp_matrix.cpp
	blockSQP/src/blocksqp_problemspec.cpp
	blockSQP/src/blocksqp_general_purpose.cpp
	blockSQP/src/blocksqp_glob.cpp
   	blockSQP/src/blocksqp_hess.cpp
   	blockSQP/src/blocksqp_iter.cpp
   	blockSQP/src/blocksqp_method.cpp
   	blockSQP/src/blocksqp_mainloop.cpp
   	blockSQP/src/blocksqp_options.cpp
   	blockSQP/src/blocksqp_qp.cpp
   	blockSQP/src/blocksqp_restoration.cpp
   	blockSQP/src/blocksqp_stats.cpp
   	blockSQP/src/blocksqp_qpsolver.cpp
   	blockSQP/src/blocksqp_defs.cpp
   	blockSQP/src/blocksqp_scaling.cpp
   	blockSQP/src/blocksqp_sqputils.cpp
	blockSQP/src/ext/blocksqp_load_mumps.cpp
   	)
target_compile_definitions(blockSQP PUBLIC QPSOLVER_QPOASES)
#target_compile_options(blockSQP PRIVATE -g -O0)
#target_compile_options(blockSQP PRIVATE -fvisibility=hidden)

if (UNIX AND NOT APPLE)
	target_compile_definitions(blockSQP PUBLIC LINUX)
elseif(WINDOWS)
	target_compile_definitions(blockSQP PUBLIC WINDOWS)
endif()

target_compile_options(blockSQP PRIVATE 
	-O3 
	#-g -O0
	-Wall -Wextra -Wno-unused-parameter)

#target_compile_options(blockSQP PRIVATE -DLINSOL_PATH=${MUMPS_BINARY_DIR}/libdmumps.so)

set_target_properties(blockSQP PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/lib)

target_include_directories(blockSQP PUBLIC blockSQP/include ${qpOASES_MODDED_DIR}/include ${qpOASES_MODDED_DIR}/include/qpOASES)
target_link_libraries(blockSQP PUBLIC qpOASES)


if (UNIX AND NOT APPLE)
	add_library(dmumps_c_dyn SHARED blockSQP/src/ext/dmumps_c_dyn.cpp)
	set_target_properties(dmumps_c_dyn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/lib/ext)
	target_link_libraries(dmumps_c_dyn PRIVATE MUMPS::MUMPS)
elseif(WIN32)
	foreach(i RANGE 0 7)
		add_library(dmumps_c_dyn_${i} SHARED blockSQP/src/ext/dmumps_c_dyn.cpp)
		set_target_properties(dmumps_c_dyn_${i} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/lib/ext)
		target_link_libraries(dmumps_c_dyn_${i} PRIVATE MUMPS::MUMPS)
	endforeach()
endif()

target_link_libraries(blockSQP PUBLIC dmumps_c_dyn)



add_executable(example1 blockSQP/examples/example1.cpp)
target_link_libraries(example1 blockSQP)
set_target_properties(example1
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP/examples")
	
add_executable(example1_sparse blockSQP/examples/example1_sparse.cpp)
target_link_libraries(example1_sparse blockSQP)
set_target_properties(example1_sparse
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP/examples")

add_executable(example_condensing blockSQP/examples/example_condensing.cpp)
target_link_libraries(example_condensing blockSQP)
target_compile_options(example_condensing PRIVATE -g -O0)
set_target_properties(example_condensing
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP/examples"
)

add_executable(example_CQPsolver blockSQP/examples/example_CQPsolver.cpp)
target_link_libraries(example_CQPsolver blockSQP)
target_compile_options(example_CQPsolver PRIVATE -g -O0)
set_target_properties(example_CQPsolver
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP/examples"
)

#[[
add_executable(example_plugin_loader blockSQP/examples/example_plugin_loader.cpp)
target_link_libraries(example_plugin_loader blockSQP)
target_compile_options(example_plugin_loader PRIVATE -g -O0)
set_target_properties(example_plugin_loader
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP/examples"
)
]]

option(PYTHON_INTERFACE "build python interface" OFF)
set(PYTHON_INTERPRETER "" CACHE STRING "The python interpreter the interface should be built for")

if(PYTHON_INTERFACE)
	if (NOT PYTHON_INTERPRETER STREQUAL "")
		message(STATUS "Using python interpreter ${PYTHON_INTERPRETER}")
		set (PYTHON_EXECUTABLE ${PYTHON_INTERPRETER})
	endif()
	
	FetchContent_Declare(
		PYBIND11
		GIT_REPOSITORY https://github.com/pybind/pybind11.git
		GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4	#2.13.6
	)
	FetchContent_MakeAvailable(PYBIND11)
	
	pybind11_add_module(py_blockSQP python_Interface/py_blockSQP.cpp)
	target_include_directories(py_blockSQP PUBLIC blockSQP/include)	
	
	target_compile_options(py_blockSQP PRIVATE 
				-fvisibility=hidden
				-Wall -Wextra -Wno-unused-parameter
				-O3
				#-g -O0
				)
	
	#Prevent warnings for *_pointer_interface and T_array constructors 
	#that occur after compiling with -O3.
	target_link_options(py_blockSQP PRIVATE -Wno-alloc-size-larger-than)
	target_link_libraries(py_blockSQP PUBLIC blockSQP)
	#set_target_properties(py_blockSQP PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python_Interface)
	set_target_properties(py_blockSQP PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python_Interface/py_blockSQP)
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
	add_library(blockSQP_jl SHARED julia_Interface/blockSQP_jl.cpp)

	target_include_directories(blockSQP_jl PUBLIC blockSQP/include)
	target_compile_options(blockSQP_jl PRIVATE -Wall -Wextra -Wno-unused-parameter -fvisibility=hidden)
	target_link_libraries(blockSQP_jl PUBLIC blockSQP JlCxx::cxxwrap_julia)
	#set_target_properties(blockSQP_jl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/julia_Interface)
	set_target_properties(blockSQP_jl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blocksqp.jl/bin)
endif()

