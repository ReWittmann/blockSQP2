cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Please use out of source build:
  cmake -B build
  {make -C build } OR {cmake --build build}")
endif()

project(BlockSQP_2    
	DESCRIPTION "Extensions and modifications for blockSQP, the nonlinear programming solver by Dennis Janka"
	HOMEPAGE_URL https://github.com/ReWittmann/blockSQP_2
    LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 20)
enable_language(Fortran)	#For MUMPS linear solver


#Static linking to avoid having to copy dll(-path)s.
if(WIN32)
	set(BLA_VENDOR OpenBLAS)
	set(LAPACK_VENDOR OpenBLAS)
	set(BLA_STATIC OFF)
	
	set(OPENBLAS_DIR "" CACHE STRING "Path to the OpenBLAS installation folder containing lib, bin and include folders. Make sure it is the 32bit-integer version (e.g. Openblas-*.*.**-x64)")
	if (OPENBLAS_DIR STREQUAL "")
		message(FATAL_ERROR "\nERROR: Please pass path to OpenBLAS via cmake -B build ... -DOPENBLAS_DIR=\"PATH/TO/OPENBLAS\". The folder should contain lib, bin and include folders. Make sure it is the 32bit-integer version (e.g. Openblas-*.*.**-x64)")
	endif()
	file(TO_CMAKE_PATH ${OPENBLAS_DIR} OPENBLAS_DIR)
	set(CMAKE_PREFIX_PATH ${OPENBLAS_DIR})
	set(OPENBLAS_INCLUDE_DIR ${OPENBLAS_DIR}/include ${OPENBLAS_DIR}/include/openblas)
endif()

find_package(BLAS REQUIRED)
message(STATUS "BLAS_FOUND = ${BLAS_FOUND}")
message(STATUS "BLAS_LIBRARIES = ${BLAS_LIBRARIES}")
find_package(LAPACK REQUIRED)
message(STATUS "LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}")
set(BLAS_LIBRARY ${BLAS_LIBRARIES})
set(LAPACK_LIBRARY ${LAPACK_LIBRARIES})

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
	set(MUMPS_find_static OFF)
	set(MUMPS_openmp OFF)
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Os")
else()
	set(MUMPS_find_static OFF)
	set(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS} -Os) #Likely to be loaded several times with dlmopen + no noticeable performance difference compared to -O3
endif()
set(MUMPS_intsize64 OFF) #We use 32bit-integer BLAS, so this must be OFF

#Default to static, since we later build a shared library from this
set(BUILD_SHARED_LIBS OFF)

FetchContent_MakeAvailable(CMAKEMUMPS)

#############################################################
#Build instuction for contained files

set(qpOASES_MODDED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/blockSQP/dep/modified_qpOASES)
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
target_link_libraries(qpOASES PUBLIC MUMPS::MUMPS LAPACK::LAPACK BLAS::BLAS) #
IF(UNIX)
	target_compile_definitions(qpOASES PRIVATE LINUX)
	target_compile_options(qpOASES PUBLIC -fPIC PRIVATE -O3 INTERFACE -Wno-overloaded-virtual)
	if (NOT APPLE)
		target_link_libraries(qpOASES PUBLIC -lgfortran)
	endif()
ELSEIF(WIN32)
    #target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
	if(MINGW)
		target_compile_options(qpOASES PRIVATE -DWIN32 
			-Wall -pedantic -Wfloat-equal -Wshadow 
			-O3 -finline-functions
		)
	endif()
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
FetchContent_MakeAvailable(qpos)

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
target_compile_options(qpOASES PUBLIC -fPIC PRIVATE -O3 INTERFACE -Wno-overloaded-virtual)
IF(UNIX AND NOT APPLE)
   target_compile_definitions(qpOASES PRIVATE LINUX)
ELSEIF(WIN32)
    target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
ENDIF()

target_include_directories(qpOASES
	PUBLIC	${qpOASES_MODDED_DIR}/include
)
# target_link_libraries(qpOASES PUBLIC /path/to/libcoinhsl.so /path/to/libmetis.a /path/to/libGKlib.a LAPACK::LAPACK BLAS::BLAS dl)
]]



#As of right now, there are no export specifications ( __declspec(dllexport), __attribute__((visibility("default"))) )
#in the code, so static linking is recommended.
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
		-Wall -Wextra -Wno-unused-parameter -Wno-maybe-uninitialized
		-O3
		# -DMATRIX_DEBUG		#Enable bounds and dimension checks for matrix operations
		)
elseif(WIN32)
	target_compile_definitions(blockSQP PUBLIC WINDOWS)
	if(MINGW)
		target_compile_options(blockSQP PRIVATE 
			-Wall -Wextra -Wno-unused-parameter -Wno-maybe-uninitialized
			-O3
			# -DMATRIX_DEBUG		#Enable bounds and dimension checks for matrix operations
			)
	endif()
endif()

set_target_properties(blockSQP PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin
										  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/lib
										  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin)

target_include_directories(blockSQP PUBLIC blockSQP/include ${qpOASES_MODDED_DIR}/include ${qpOASES_MODDED_DIR}/include/qpOASES ${OPENBLAS_INCLUDE_DIR})
target_link_libraries(blockSQP PUBLIC qpOASES) #Important: If qpOASES is built with MUMPS, -DSOLVER_MUMPS must be inherited
											   #		   to enable workarounds for MUMPS not being thread safe,
											   #		   or prevent MUMPS from running in parallel

#Shared mumps library/libraries for dynamic loading in a threadsafe way (dlmopen on linux, LoadLibraryA on multiple copies on windows, tough luck on mac)
#Also attach copying of dlls here since we copy dmumps_c_dyn anyways.
set(COPY_COUNTER 0)
if (UNIX AND NOT APPLE)
	add_library(dmumps_c_dyn SHARED blockSQP/src/ext/dmumps_c_dyn.cpp)
	set_target_properties(dmumps_c_dyn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin
												  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/lib
												  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin)
	target_link_libraries(dmumps_c_dyn PRIVATE MUMPS::MUMPS)
	target_compile_options(dmumps_c_dyn PRIVATE -O3 -fvisibility=hidden)

	function(copy_dynamic_libraries_to LIBPATH)
		#[[
		set(mumps_lib dmumps_c_dyn)
		add_custom_command(TARGET ${mumps_lib} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
					${CMAKE_SOURCE_DIR}/blockSQP/bin/lib${mumps_lib}.so
					${LIBPATH}
		)
		]]

		set(MUMPS_COPY_TARGET MCT_${COPY_COUNTER})
		MATH(EXPR COPY_COUNTER_LOCAL "${COPY_COUNTER} + 1")
		set(COPY_COUNTER ${COPY_COUNTER_LOCAL} PARENT_SCOPE)

		add_custom_target(${MUMPS_COPY_TARGET} ALL
			COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/blockSQP/bin/libdmumps_c_dyn.so ${LIBPATH}
			COMMENT "Copying dynamic libraries\n"
		)

		add_dependencies(${MUMPS_COPY_TARGET} dmumps_c_dyn)
	endfunction()
elseif(WIN32)	
	add_library(dmumps_c_dyn SHARED blockSQP/src/ext/dmumps_c_dyn.cpp)
	set_target_properties(dmumps_c_dyn PROPERTIES 
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin
		ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/lib
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP/bin)
	target_link_libraries(dmumps_c_dyn PRIVATE MUMPS::MUMPS)
	
	if(MINGW)
		target_link_libraries(dmumps_c_dyn PRIVATE -static)
		target_compile_options(dmumps_c_dyn PRIVATE -O3 -fvisibility=hidden)
	endif()
	target_compile_definitions(dmumps_c_dyn PRIVATE WINDOWS)

	#In case of dynamic linking of libgfortran, libwinpthread, libstdc++ etc., search for them in the gcc folder
	#Save them in a variable so they get copied to executable requiring them
	#[[
	execute_process(
		COMMAND where gcc.exe
		OUTPUT_VARIABLE GCC_PATH
		RESULT_VARIABLE GCC_ERROR
		OUTPUT_STRIP_TRAILING_WHITESPACE
	)
	if (GCC_ERROR)
		message(FATAL_ERROR "gcc.exe not found on PATH!")
	endif()

	string(REPLACE "\r" "" GCC_PATH "${GCC_PATH}")
	string(REGEX MATCH "^[^\n]*" GCC_PATH "${GCC_PATH}")

	message(STATUS "Found gcc at: ${GCC_PATH}.")
	get_filename_component(GCC_BIN_DIR "${GCC_PATH}" DIRECTORY)
	
	if(NOT EXISTS ${GCC_BIN_DIR}/libstdc++-6.dll OR NOT EXISTS ${GCC_BIN_DIR}/libgfortran-5.dll OR NOT EXISTS ${GCC_BIN_DIR}/libgcc_s_seh-1.dll OR NOT EXISTS ${GCC_BIN_DIR}/libwinpthread-1.dll OR NOT EXISTS ${GCC_BIN_DIR}/libquadmath-0.dll)
		message(FATAL_ERROR "Could not find one of libstdc++-6.dll, libgfortran.dll, libgcc_s_seh-1.dll, libwinpthread-1.dll, libquadmath-0.dll in ${GCC_BIN_DIR}")
	endif()
	set(STDLIB_BINARIES ${GCC_BIN_DIR}/libstdc++-6.dll ${GCC_BIN_DIR}/libgfortran-5.dll ${GCC_BIN_DIR}/libgcc_s_seh-1.dll ${GCC_BIN_DIR}/libwinpthread-1.dll ${GCC_BIN_DIR}/libquadmath-0.dll)
	]]

	set(OPENBLAS_BINARY ${OPENBLAS_DIR}/bin/libopenblas.dll)
	if(NOT EXISTS ${OPENBLAS_BINARY})
		message(FATAL_ERROR "Expected libopenblas.dll in OPENBLAS_DIR/bin")
	endif()

	function(copy_dynamic_libraries_to LIBPATH)
		set(COPY_COMMANDS)

		##Mumps libraries for dynamic loading
		foreach(i RANGE 0 7)
			set(MUMPS_COPY_PRODUCT ${LIBPATH}/dmumps_c_dyn_${i}.dll)
			list(APPEND COPY_COMMANDS 
				COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/blockSQP/bin/libdmumps_c_dyn.dll ${MUMPS_COPY_PRODUCT}
			)
		endforeach()

		##OpenBLAS
		list(APPEND COPY_COMMANDS
				COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENBLAS_DIR}/bin/libopenblas.dll ${LIBPATH}
			)
		
		##C/C++/Fortran etc. standard libraries
		if (STDLIB_BINARIES)
			list(APPEND COPY_COMMANDS
				COMMAND ${CMAKE_COMMAND} -E copy_if_different ${STDLIB_BINARIES} ${LIBPATH}
			)
		endif()

		set(MUMPS_COPY_TARGET MCT_${COPY_COUNTER})
		MATH(EXPR COPY_COUNTER_LOCAL "${COPY_COUNTER} + 1")
		set(COPY_COUNTER ${COPY_COUNTER_LOCAL} PARENT_SCOPE)

		add_custom_target(${MUMPS_COPY_TARGET} ALL
						  ${COPY_COMMANDS}
						  COMMENT "Copying dynamic libraries\n"
		)

		add_dependencies(${MUMPS_COPY_TARGET} dmumps_c_dyn)
	endfunction()
else()
	function(copy_dynamic_libraries_to LIBPATH)
	endfunction()
endif()

#Mumps dynamic libraries are available. Set corresponding preprocessor flag
target_compile_definitions(blockSQP PRIVATE DMUMPS_C_DYN)

#Build examples and place executables and required libraries in dedicated folder
copy_dynamic_libraries_to(${CMAKE_SOURCE_DIR}/blockSQP/examples/bin)

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


option(PYTHON_INTERFACE "build python interface" ON)
set(PYTHON_INTERPRETER "" CACHE STRING "The python interpreter the interface should be built for")
if(PYTHON_INTERFACE)
	if (NOT PYTHON_INTERPRETER STREQUAL "")
		message(STATUS "Using python interpreter ${PYTHON_INTERPRETER}")
		set (PYTHON_EXECUTABLE ${PYTHON_INTERPRETER})
	endif()
	
	FetchContent_Declare(
		PYBIND11
		GIT_REPOSITORY https://github.com/pybind/pybind11.git
		GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4	#2.13.6; feel free to update to a later version
	)
	FetchContent_MakeAvailable(PYBIND11)
	
	pybind11_add_module(py_blockSQP py_blockSQP/py_blockSQP.cpp)
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
	elseif(MINGW)
		target_compile_options(py_blockSQP PRIVATE 
					-Wall -Wextra -Wno-unused-parameter
					-fvisibility=hidden
					-O3
					)
		target_link_options(py_blockSQP PRIVATE -Wno-alloc-size-larger-than)
	endif()
	
	target_link_libraries(py_blockSQP PUBLIC blockSQP)

	if(MINGW)
		target_link_libraries(py_blockSQP PRIVATE -static)
	endif()

	set(py_blockSQP_DIR ${CMAKE_SOURCE_DIR}/py_blockSQP)
	set_target_properties(py_blockSQP PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${py_blockSQP_DIR}
												 RUNTIME_OUTPUT_DIRECTORY ${py_blockSQP_DIR}
												 )
	copy_dynamic_libraries_to(${py_blockSQP_DIR})
endif()


#Build C interface to complete blockSQP.jl
option(JULIA_INTERFACE "build Julia interface" ON)
if(JULIA_INTERFACE)
	add_library(blockSQP_jl SHARED C_blockSQP/C_blockSQP.cpp)
	target_link_libraries(blockSQP_jl PUBLIC blockSQP)
	
	target_include_directories(blockSQP_jl PUBLIC blockSQP/include)
	if(UNIX)
		target_compile_options(blockSQP_jl PRIVATE 
					-Wall -Wextra -Wno-unused-parameter
					-fvisibility=hidden
					-O3
					)
	elseif(MINGW)
		target_compile_options(blockSQP_jl PRIVATE 
					-Wall -Wextra -Wno-unused-parameter
					-fvisibility=hidden
					-O3
					)
		target_link_libraries(blockSQP_jl PRIVATE -static)
	endif()
	set_target_properties(blockSQP_jl PROPERTIES 
			PREFIX lib
			LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP.jl/bin
			RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP.jl/bin
	)
	copy_dynamic_libraries_to(${CMAKE_SOURCE_DIR}/blockSQP.jl/bin)
endif()

