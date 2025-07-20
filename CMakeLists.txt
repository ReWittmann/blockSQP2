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
		message(FATAL_ERROR "This CMakeLists support building on windows only with MSVC and intel oneAPI fortran")
	endif()
	set(INTEL_STATIC_LINK_FLAGS "-static-intel")
	add_compile_options(/MT)
endif()

include(FetchContent)
FetchContent_Declare(
	CMAKEMUMPS
	GIT_REPOSITORY https://github.com/scivision/mumps.git
	GIT_TAG 45c16604c1d8b61134699db04b16aae7b4704fc5
)
set(MUMPS_parallel OFF)
set(MUMPS_scalapack OFF)
set(BUILD_SINGLE OFF)
set(MUMPS_find_static ON)

#Shared MUMPS libraries are not supported when building with oneAPI and MSVC on windows
# + a separate shared library may need to be built from this, so default to static
set(BUILD_SHARED_LIBS OFF)

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
ELSEIF(WIN32)
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




#As of right now, there are to export specifications ( __declspec((dllexport)), __attribute__((visibility("default"))) )
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

#Shared mumps library/ies for dynamic loading in a threadsafe way (dlmopen on linux, LoadLibraryA on multiple copies on windows)
if (UNIX AND NOT APPLE)
	add_library(dmumps_c_dyn SHARED blockSQP/src/ext/dmumps_c_dyn.cpp)
	set_target_properties(dmumps_c_dyn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin
												ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/lib
												RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin)
	target_link_libraries(dmumps_c_dyn PRIVATE MUMPS::MUMPS)
	target_compile_definitions(dmumps_c_dyn INTERFACE LDMUMPS_C_DYN)
	target_link_libraries(blockSQP PUBLIC dmumps_c_dyn)
	
	function(ifwindows_copy_mumps_dlls LIBPATH)
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
	
	function(ifwindows_copy_mumps_dlls LIBPATH)
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
	function(ifwindows_copy_mumps_dlls LIBPATH)
	endfunction()
endif()


ifwindows_copy_mumps_dlls(${CMAKE_SOURCE_DIR}/blockSQP/examples/bin)

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
# target_compile_options(example_condensing PRIVATE -g -O0)
set_target_properties(example_condensing
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP/examples/bin"
)

add_executable(example_CQPsolver blockSQP/examples/example_CQPsolver.cpp)
target_link_libraries(example_CQPsolver blockSQP)
# target_compile_options(example_CQPsolver PRIVATE -g -O0)
set_target_properties(example_CQPsolver
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
		#that occur after compiling with -O3. Result of using int instead of size_t.
		target_link_options(py_blockSQP PRIVATE -Wno-alloc-size-larger-than)
	endif()
	
	target_link_libraries(py_blockSQP PUBLIC blockSQP)
	set_target_properties(py_blockSQP PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python_Interface/py_blockSQP
												 RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python_Interface/py_blockSQP
												 )
	ifwindows_copy_mumps_dlls(${CMAKE_SOURCE_DIR}/python_Interface/py_blockSQP)
endif()


option(JULIA_INTERFACE "build julia interface" OFF)
set(CXXWRAP_PATH "" CACHE STRING "Path to CxxWrap artifact libcxxwrap-julia (default: empty)")

if(JULIA_INTERFACE)

	
	#[[
	set(JLCXX_BUILD_EXAMPLES OFF)
	set(JLCXX_BUILD_TESTS OFF)
	FetchContent_Declare(
		JlCxx
		GIT_REPOSITORY https://github.com/JuliaInterop/libcxxwrap-julia.git
		GIT_TAG 3565a1afa4fa18c683bf02ec76f7b7d21b93bfcd #0.14.3
	)
	FetchContent_MakeAvailable(JlCxx)
	
	message(STATUS "JlCxx include dir: " ${JlCxx_SOURCE_DIR}/include)
	set(LibCxxWrap_INCLUDE_DIR ${JlCxx_SOURCE_DIR}/include)
	if (WIN32)
		set(LibCxxWrap_BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
	else()
		set(LibCxxWrap_BINARY_DIR ${CMAKE_BINARY_DIR}/lib)
	endif()
	message(STATUS "JlCxx bin dir: " ${LibCxxWrap_BINARY_DIR})
	
	
	
	#file(REMOVE_RECURSE ${LIBCXX_OVERRIDE_DIR})
	execute_process( #TODO possibly add Pkg.free(\"libcxxwrap_julia_jll\")
		COMMAND julia -E "begin print(joinpath.(DEPOT_PATH)[1]); exit() end"
		OUTPUT_VARIABLE dotJULIA_DIR
		OUTPUT_STRIP_TRAILING_WHITESPACE
	)
	message(STATUS "Julia path: " ${dotJULIA_DIR})
	
	###NEW###
	execute_process(
		COMMAND julia -E "begin import Pkg; Pkg.activate(\"${CMAKE_SOURCE_DIR}/blocksqp.jl\"); Pkg.instantiate(); exit() end"
		OUTPUT_VARIABLE dotJULIA_DIR_TEST
		OUTPUT_STRIP_TRAILING_WHITESPACE
	)

	execute_process(
		COMMAND julia --project=${CMAKE_SOURCE_DIR}/blocksqp.jl -E "begin import Pkg; Pkg.develop(\"libcxxwrap_julia_jll\"); import libcxxwrap_julia_jll; libcxxwrap_julia_jll.dev_jll(); print(joinpath.(DEPOT_PATH)[1]); exit() end"
		OUTPUT_VARIABLE dotJULIA_DIR_TEST
		OUTPUT_STRIP_TRAILING_WHITESPACE
	)
	#########
	]]
	
	
	#[[
	execute_process( #TODO possibly add Pkg.free(\"libcxxwrap_julia_jll\")
		COMMAND julia -E "begin import Pkg; Pkg.develop(\"libcxxwrap_julia_jll\"); import libcxxwrap_julia_jll; libcxxwrap_julia_jll.dev_jll(); print(joinpath.(DEPOT_PATH)[1]); exit() end"
		OUTPUT_VARIABLE dotJULIA_DIR_TEST
		OUTPUT_STRIP_TRAILING_WHITESPACE
	)
	]]
	
	
	#[[
	if("${dotJULIA_DIR_TEST}" STREQUAL "")
		message(FATAL_ERROR "Error in julia environment")
	endif()
	
	set(LIBCXX_OVERRIDE_DIR ${dotJULIA_DIR}/dev/libcxxwrap_julia_jll/override)
	
	if(NOT EXISTS ${LIBCXX_OVERRIDE_DIR}_save_CMAKE_blockSQP)
		file(RENAME ${LIBCXX_OVERRIDE_DIR} ${LIBCXX_OVERRIDE_DIR}_save_CMAKE_blockSQP)
	else()
		file(REMOVE_RECURSE ${LIBCXX_OVERRIDE_DIR})
	endif()
	file(MAKE_DIRECTORY ${LIBCXX_OVERRIDE_DIR})
	
	
	message(STATUS "LibCxxWrap_BINARY_DIR" ${LibCxxWrap_BINARY_DIR})
	message(STATUS "LibCxxWrap_INCLUDE_DIR" ${LibCxxWrap_INCLUDE_DIR})
	
	add_custom_target(copy_to_override ALL
    	COMMAND ${CMAKE_COMMAND} -E copy_directory ${LibCxxWrap_BINARY_DIR} ${LIBCXX_OVERRIDE_DIR}/lib
    	COMMAND ${CMAKE_COMMAND} -E copy_directory ${LibCxxWrap_INCLUDE_DIR} ${LIBCXX_OVERRIDE_DIR}/include
	)
	add_dependencies(copy_to_override cxxwrap_julia)
	]]
	
	add_library(blockSQP_jl SHARED julia_Interface/blockSQP_jl_NOCXXWRAP.cpp)
	target_link_libraries(blockSQP_jl PUBLIC blockSQP)
	
	target_include_directories(blockSQP_jl PUBLIC blockSQP/include)# ${JlCxx_SOURCE_DIR}/include)
	if(UNIX)
		target_compile_options(blockSQP_jl PRIVATE -Wall -Wextra -Wno-unused-parameter -fvisibility=hidden)
	endif()
	set_target_properties(blockSQP_jl PROPERTIES 
			LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blocksqp.jl/bin
			RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blocksqp.jl/bin
	)
	ifwindows_copy_mumps_dlls(${CMAKE_SOURCE_DIR}/blocksqp.jl/bin)
	
	
	
	#[[
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
	
	if(UNIX)
		target_compile_options(blockSQP_jl PRIVATE -Wall -Wextra -Wno-unused-parameter)
		#DONT ADD -fvisibility=hidden, else ERROR: LoadError: No appropriate factory for type v
	endif()
	
	target_link_libraries(blockSQP_jl PUBLIC blockSQP JlCxx::cxxwrap_julia)
	set_target_properties(blockSQP_jl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blocksqp.jl/bin)
	
	ifwindows_copy_mumps_dlls(${CMAKE_SOURCE_DIR}/blocksqp.jl/bin)
	]]
endif()

