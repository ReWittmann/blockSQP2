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

#Static linking to avoid having to copy dll(-path)s.
#
if(WIN32)
	if(NOT MSVC)
		message(FATAL_ERROR "This CMakeLists supports building on windows only with MSVC and intel oneAPI fortran")
	endif()
	set(INTEL_STATIC_LINK_FLAGS "-static-intel")
	add_compile_options(/MT)
endif()

include(FetchContent)
FetchContent_Declare(
	CMAKEMUMPS
	GIT_REPOSITORY https://github.com/scivision/mumps.git
	GIT_TAG 5b153379cdd1b0d2a33daf88bffab16a22ff0004
)
set(MUMPS_parallel OFF)
set(MUMPS_scalapack OFF)
set(BUILD_SINGLE OFF)


if(WIN32)
	set(MUMPS_find_static ON)
else()
	set(MUMPS_find_static OFF)
	set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} -Os)
endif()
set(MUMPS_intsize64 OFF) #must be off

#Shared MUMPS libraries are not supported when building with oneAPI and MSVC on windows
# + a separate shared library may need to be built from this, so default to static
set(BUILD_SHARED_LIBS OFF)

FetchContent_MakeAvailable(CMAKEMUMPS)
#############################################################

set(qpOASES_MODDED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/modded_external/qpOASES_mod)
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
target_compile_definitions(qpOASES PUBLIC SOLVER_MUMPS PRIVATE MUMPS_SEQ __NO_COPYRIGHT__)
target_link_libraries(qpOASES PUBLIC MUMPS::MUMPS) #LAPACK and BLAS dependencies are already resolved by CMakeMumps and passed on
IF(UNIX)
	target_compile_definitions(qpOASES PRIVATE LINUX)
	target_compile_options(qpOASES PUBLIC -fPIC PRIVATE -O3 INTERFACE -Wno-overloaded-virtual)
	if (NOT APPLE)
		target_link_libraries(qpOASES PUBLIC -lgfortran)
	endif()
ELSEIF(WIN32)
    target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
ENDIF()
target_include_directories(qpOASES
	PUBLIC	${qpOASES_MODDED_DIR}/include
)

target_include_directories(qpOASES
	PUBLIC	${qpOASES_MODDED_DIR}/include ${qpOASES_MODDED_DIR}/include/qpOASES
)



#Unmodded qpOASES with MUMPS_SPARSE_SOLVER fix
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
	${CMAKE_CURRENT_SOURCE_DIR}/modded_external/SparseSolver.cpp
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
target_compile_definitions(qpOASES PUBLIC SOLVER_MUMPS PRIVATE MUMPS_SEQ __NO_COPYRIGHT__)
target_compile_options(qpOASES PRIVATE -O3 INTERFACE -Wno-overloaded-virtual PUBLIC -fPIC)
target_link_libraries(qpOASES PUBLIC MUMPS::MUMPS)
IF(UNIX AND NOT APPLE)
   target_compile_definitions(qpOASES PRIVATE LINUX)
		target_link_libraries(qpOASES PUBLIC -lgfortran)
ELSEIF(WIN32)
    target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
ENDIF()

target_include_directories(qpOASES
	PUBLIC	${qpos_SOURCE_DIR}/include
			${qpos_SOURCE_DIR}/include/qpOASES
)
]]



#MA57 LINEAR SOLVER
## Build qpOASES with MA57, requires passing preprocessor flag -DSOLVER_MA57
## Dependencies are -> METIS -> GKLIB, LAPACK, BLAS, dl
#[[
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_compile_definitions(qpOASES PRIVATE SOLVER_MA57 __NO_COPYRIGHT__)
target_compile_options(qpOASES PRIVATE -O3)
IF(UNIX AND NOT APPLE)
   target_compile_definitions(qpOASES PRIVATE LINUX)
ELSEIF(WIN32)
    target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
ENDIF()

target_include_directories(qpOASES
	PUBLIC	${qpOASES_MODDED_DIR}/include
)
target_link_libraries(qpOASES PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libcoinhsl.so /path/to/libmetis.a /path/to/libGKlib.a LAPACK::LAPACK BLAS::BLAS dl)
]]


#SPRAL LINEAR SOLVER
## Requires setting environment variables export OMP_CANCELLATION=TRUE and OMP_PROC_BIND=TRUE before every run, see SRPAL readme
#[[
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_compile_definitions(qpOASES PRIVATE SOLVER_SPRAL __NO_COPYRIGHT__)
target_compile_options(qpOASES PRIVATE -O3)
IF(UNIX AND NOT APPLE)
   target_compile_definitions(qpOASES PRIVATE LINUX)
ELSEIF(WIN32)
    target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
ENDIF()

target_include_directories(qpOASES
	PUBLIC	${qpOASES_MODDED_DIR}/include /path/to/.../spral/include
)
target_link_libraries(qpOASES PUBLIC /path/to/.../libspral.so dl)
]]


#As of right now, there are no export specifications ( __declspec((dllexport)), __attribute__((visibility("default"))) )
#in the code, so static linking is recommended
add_library(blockSQP STATIC
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
	blockSQP/src/blocksqp_load_mumps.cpp
   	)
target_compile_definitions(blockSQP PUBLIC QPSOLVER_QPOASES)


if (UNIX)
	if(NOT APPLE)
		target_compile_definitions(blockSQP PUBLIC LINUX)
	endif()
	target_compile_options(blockSQP PRIVATE 
		-O3
		-Wall -Wextra -Wno-unused-parameter -Wno-maybe-uninitialized)
elseif(WIN32)
	target_compile_definitions(blockSQP PUBLIC WINDOWS)
endif()


set_target_properties(blockSQP PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin
										  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/lib
										  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin)

target_include_directories(blockSQP PUBLIC blockSQP/include ${qpOASES_MODDED_DIR}/include ${qpOASES_MODDED_DIR}/include/qpOASES)
target_link_libraries(blockSQP PUBLIC qpOASES) #Important: If qpOASES is built with MUMPS, -DSOLVER_MUMPS must be inherited
											   #		   to enable workarounds for MUMPS not being thread safe,
											   #		   or prevent MUMPS from running in parallel

#Shared mumps library/libraries for dynamic loading in a threadsafe way (dlmopen on linux, LoadLibraryA on multiple copies on windows, tough luck if you are on MAC)
if (UNIX AND NOT APPLE)
	add_library(dmumps_c_dyn SHARED blockSQP/src/ext/dmumps_c_dyn.cpp)
	set_target_properties(dmumps_c_dyn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin
												ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/lib
												RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin)
	target_link_libraries(dmumps_c_dyn PRIVATE MUMPS::MUMPS)
	target_compile_options(dmumps_c_dyn PRIVATE -O3 -fvisibility=hidden)
	#Notify downstream libraries that dmumps_c_dyn will be made available through LDMUMPS_C_DYN compile flag
	#As of right now, this is only used on linux
	# target_compile_definitions(dmumps_c_dyn INTERFACE DMUMPS_C_DYN)
	# target_link_libraries(blockSQP PUBLIC dmumps_c_dyn)
	
	
	function(maybe_copy_mumps_libs LIBPATH)
		set(mumps_lib dmumps_c_dyn)
		add_custom_command(TARGET ${mumps_lib} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
					${CMAKE_SOURCE_DIR}/blockSQP/bin/lib${mumps_lib}.so
					${LIBPATH}
		)
	endfunction()
elseif(WIN32)
	foreach(i RANGE 0 7)
		add_library(dmumps_c_dyn_${i} SHARED blockSQP/src/ext/dmumps_c_dyn.cpp)
		set_target_properties(dmumps_c_dyn_${i} PROPERTIES 
				LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin
				ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/lib
				RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin)
		target_link_libraries(dmumps_c_dyn_${i} PRIVATE MUMPS::MUMPS)
		target_compile_definitions(dmumps_c_dyn_${i} PRIVATE WINDOWS)
	endforeach()
	
	function(maybe_copy_mumps_libs LIBPATH)
		foreach(i RANGE 0 7)
			set(mumps_lib dmumps_c_dyn_${i})
			add_custom_command(TARGET ${mumps_lib} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
						${CMAKE_SOURCE_DIR}/blockSQP/bin/${mumps_lib}.dll
						${LIBPATH}
			)
		endforeach()
	endfunction()
else()
	function(maybe_copy_mumps_libs LIBPATH)
	endfunction()
endif()

#Mumps dynamic libraries are available. Set corresponding preprocessor flag
target_compile_definitions(blockSQP PRIVATE DMUMPS_C_DYN)
maybe_copy_mumps_libs(${CMAKE_SOURCE_DIR}/blockSQP/examples/bin)

add_executable(example1 blockSQP/examples/example1.cpp)
target_link_libraries(example1 blockSQP)
set_target_properties(example1
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP/examples/bin")
	
add_executable(example1_sparse blockSQP/examples/example1_sparse.cpp)
target_link_libraries(example1_sparse blockSQP)
set_target_properties(example1_sparse
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP/examples/bin")

add_executable(example_condensing blockSQP/examples/example_condensing.cpp)
target_link_libraries(example_condensing blockSQP)
set_target_properties(example_condensing
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP/examples/bin"
)


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
	
	if(UNIX)
		target_compile_options(py_blockSQP PRIVATE 
					-Wall -Wextra -Wno-unused-parameter
					-fvisibility=hidden #Recommended in pybind11 FAQ, greatly reduces binary size
					-O3
					)
		
		#Prevent warnings for *_pointer_interface and T_array constructors 
		#that occur after compiling with -O3. Result of using int instead of size_t, should not cause any problems.
		target_link_options(py_blockSQP PRIVATE -Wno-alloc-size-larger-than)
	endif()
	
	target_link_libraries(py_blockSQP PUBLIC blockSQP)
	set_target_properties(py_blockSQP PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python_Interface/py_blockSQP
												 RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python_Interface/py_blockSQP
												 )
	maybe_copy_mumps_libs(${CMAKE_SOURCE_DIR}/python_Interface/py_blockSQP)
endif()


option(JULIA_INTERFACE "build julia interface" OFF)
set(CXXWRAP_PATH "" CACHE STRING "Path to CxxWrap artifact libcxxwrap-julia (default: empty)")

if(JULIA_INTERFACE)
	add_library(blockSQP_jl SHARED C_Interface/C_blockSQP.cpp)
	target_link_libraries(blockSQP_jl PUBLIC blockSQP)
	
	target_include_directories(blockSQP_jl PUBLIC blockSQP/include)
	if(UNIX)
		target_compile_options(blockSQP_jl PRIVATE 
					-Wall -Wextra -Wno-unused-parameter
					-fvisibility=hidden #Recommended in pybind11 FAQ, greatly reduces binary size
					-O3
					)
	endif()
	set_target_properties(blockSQP_jl PROPERTIES 
			PREFIX lib
			LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blocksqp.jl/bin
			RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blocksqp.jl/bin
	)
	maybe_copy_mumps_libs(${CMAKE_SOURCE_DIR}/blocksqp.jl/bin)
endif()

