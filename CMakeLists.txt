cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Please use out of source build:
  cmake -B build
  {make -C build } OR {cmake --build build}")
endif()

project(CMakeblockSQP2
	DESCRIPTION "Build system for blockSQP2, a nonlinear programming solver based on blockSQP by Dennis Janka."
	HOMEPAGE_URL https://github.com/ReWittmann/blockSQP2
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
	SCIV_MUMPS_SUPERBUILD
	GIT_REPOSITORY https://github.com/scivision/mumps-superbuild.git
	GIT_TAG df62d7630760f3efd608f19c8e2002542897f7a9  # v5.8.2.1
)

# set(gemmt OFF)
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
set(MUMPS_intsize64 OFF) #We use 32bit-integer BLAS (LP64), so this must be OFF

#Default to static, since we later build a shared library from this
set(BUILD_SHARED_LIBS OFF)

FetchContent_MakeAvailable(SCIV_MUMPS_SUPERBUILD)

#############################################################
#Build instuction for contained files

set(qpOASES_MODDED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/blockSQP2/dep/modified_qpOASES)
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
target_compile_definitions(qpOASES PUBLIC SOLVER_MUMPS PRIVATE __NO_COPYRIGHT__)
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
			INTERFACE -Wno-overloaded-virtual
		)
	endif()
ENDIF()

target_include_directories(qpOASES
	PUBLIC	${qpOASES_MODDED_DIR}/include ${qpOASES_MODDED_DIR}/include/qpOASES
			${OPENBLAS_INCLUDE_DIR}
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
add_library(blockSQP2 STATIC
	blockSQP2/src/condensing.cpp
	blockSQP2/src/matrix.cpp
	blockSQP2/src/problemspec.cpp
	blockSQP2/src/general_purpose.cpp
	blockSQP2/src/glob.cpp
   	blockSQP2/src/hess.cpp
   	blockSQP2/src/iter.cpp
   	blockSQP2/src/method.cpp
   	blockSQP2/src/mainloop.cpp
   	blockSQP2/src/options.cpp
   	blockSQP2/src/qp.cpp
   	blockSQP2/src/restoration.cpp
   	blockSQP2/src/stats.cpp
   	blockSQP2/src/qpsolver.cpp
   	blockSQP2/src/defs.cpp
   	blockSQP2/src/scaling.cpp
   	blockSQP2/src/sqputils.cpp
	blockSQP2/src/load_mumps.cpp
   	)
target_compile_definitions(blockSQP2 PUBLIC QPSOLVER_QPOASES)


if (UNIX)
	if(NOT APPLE)
		target_compile_definitions(blockSQP2 PUBLIC LINUX)
	endif()
	target_compile_options(blockSQP2 PRIVATE 
		-Wall -Wextra -Wno-unused-parameter -Wno-maybe-uninitialized
		-O3
		# -DMATRIX_DEBUG		#Enable bounds and dimension checks for matrix operations
		# -DMATRIX_DEBUG -O0 -g
		)
elseif(WIN32)
	target_compile_definitions(blockSQP2 PUBLIC WINDOWS)
	if(MINGW)
		target_compile_options(blockSQP2 PRIVATE 
			-Wall -Wextra -Wno-unused-parameter -Wno-maybe-uninitialized
			-O3
			# -DMATRIX_DEBUG		#Enable bounds and dimension checks for matrix operations
			)
	endif()
endif()

set_target_properties(blockSQP2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/bin
										  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/lib
										  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/bin)

target_include_directories(blockSQP2 PUBLIC blockSQP2/include ${qpOASES_MODDED_DIR}/include)# ${OPENBLAS_INCLUDE_DIR})
target_link_libraries(blockSQP2 PUBLIC qpOASES) #Important: If qpOASES is built with MUMPS, -DSOLVER_MUMPS must be inherited
											   #		   to enable workarounds for MUMPS not being thread safe,
											   #		   or prevent MUMPS from running in parallel

#Shared mumps library/libraries for dynamic loading in a threadsafe way (dlmopen on linux, LoadLibraryA on multiple copies on windows, tough luck on mac)
#Also attach copying of dlls here since we copy dmumps_c_dyn anyways.
set(COPY_COUNTER 0)
if (UNIX AND NOT APPLE)
	add_library(dmumps_c_dyn SHARED blockSQP2/src/ext/dmumps_c_dyn.cpp)
	set_target_properties(dmumps_c_dyn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/bin
												  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/lib
												  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/bin)
	target_link_libraries(dmumps_c_dyn PRIVATE MUMPS::MUMPS)
	target_compile_options(dmumps_c_dyn PRIVATE -O3 -fvisibility=hidden)

	function(copy_dynamic_libraries_to LIBPATH)
		set(MUMPS_COPY_TARGET MCT_${COPY_COUNTER})
		MATH(EXPR COPY_COUNTER_LOCAL "${COPY_COUNTER} + 1")
		set(COPY_COUNTER ${COPY_COUNTER_LOCAL} PARENT_SCOPE)

		add_custom_target(${MUMPS_COPY_TARGET} ALL
			COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/blockSQP2/bin/libdmumps_c_dyn.so ${LIBPATH}
			COMMENT "Copying dynamic libraries\n"
		)

		add_dependencies(${MUMPS_COPY_TARGET} dmumps_c_dyn)
	endfunction()
elseif(WIN32)
	add_library(dmumps_c_dyn SHARED blockSQP2/src/ext/dmumps_c_dyn.cpp)
	set_target_properties(dmumps_c_dyn PROPERTIES 
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/bin
		ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/lib
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/bin)
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
	set(STDLIB_BINARIES ${GCC_BIN_DIR}/libstdc++-6.dll ${GCC_BIN_DIR}/libgfortran-5.dll ${GCC_BIN_DIR}/libgcc_s_seh-1.dll ${GCC_BIN_DIR}/libwinpthread-1.dll ${GCC_BIN_DIR}/libquadmath-0.dll ${GCC_BIN_DIR}/libgomp-1.dll ${GCC_BIN_DIR}/libgfortran-5.dll)
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
				COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/blockSQP2/bin/libdmumps_c_dyn.dll ${MUMPS_COPY_PRODUCT}
			)
		endforeach()

		##OpenBLAS
		list(APPEND COPY_COMMANDS
				COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENBLAS_DIR}/bin/libopenblas.dll ${LIBPATH}
			)
		
		##C/C++/Fortran etc. standard libraries
		#[[
		if (STDLIB_BINARIES)
			list(APPEND COPY_COMMANDS
				COMMAND ${CMAKE_COMMAND} -E copy_if_different ${STDLIB_BINARIES} ${LIBPATH}
			)
		endif()
		]]

		set(MUMPS_COPY_TARGET MCT_${COPY_COUNTER})
		MATH(EXPR COPY_COUNTER_LOCAL "${COPY_COUNTER} + 1")
		set(COPY_COUNTER ${COPY_COUNTER_LOCAL} PARENT_SCOPE)

		add_custom_target(${MUMPS_COPY_TARGET} ALL
						  ${COPY_COMMANDS}
						  COMMENT "Copying dynamic libraries\n"
		)

		add_dependencies(${MUMPS_COPY_TARGET} dmumps_c_dyn)
	endfunction()


	if(MINGW)
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
		set(MINGW_BINARIES ${GCC_BIN_DIR}/libstdc++-6.dll ${GCC_BIN_DIR}/libgfortran-5.dll ${GCC_BIN_DIR}/libgcc_s_seh-1.dll ${GCC_BIN_DIR}/libwinpthread-1.dll ${GCC_BIN_DIR}/libquadmath-0.dll ${GCC_BIN_DIR}/libgomp-1.dll ${GCC_BIN_DIR}/libgfortran-5.dll)
		message(STATUS "Found MINGW binaries: " ${MINGW_BINARIES})

		function(copy_mingw_libraries_to LIBPATH)
			set(MINGW_COPY_TARGET MINGW_COPY_${COPY_COUNTER})
			MATH(EXPR COPY_COUNTER_LOCAL "${COPY_COUNTER} + 1")
			set(COPY_COUNTER ${COPY_COUNTER_LOCAL} PARENT_SCOPE)
			set(MINGW_COPY_COMMAND COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MINGW_BINARIES} ${LIBPATH})

			add_custom_target(${MINGW_COPY_TARGET} ALL
							  ${MINGW_COPY_COMMAND}
							  COMMENT "Copying mingw dynamic standard libraries\n"
			)
			add_dependencies(${MINGW_COPY_TARGET} dmumps_c_dyn)
		endfunction()
	else()
		function(copy_mingw_libraries_to LIBPATH)
		endfunction()
	endif()
else()
	function(copy_dynamic_libraries_to LIBPATH)
	endfunction()

	function(copy_mingw_libraries_to LIBPATH)
	endfunction()
endif()

#Mumps dynamic libraries are available. Set corresponding preprocessor flag
target_compile_definitions(blockSQP2 PRIVATE DMUMPS_C_DYN) # -DBSQP_CBLAS_SUFFIX= 

#Build examples and place executables and required libraries in dedicated folder
copy_dynamic_libraries_to(${CMAKE_SOURCE_DIR}/blockSQP2/examples/bin)

add_executable(example1 blockSQP2/examples/example1.cpp)
target_link_libraries(example1 blockSQP2)
set_target_properties(example1
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP2/examples/bin")
	
add_executable(example1_sparse blockSQP2/examples/example1_sparse.cpp)
target_link_libraries(example1_sparse blockSQP2)
set_target_properties(example1_sparse
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP2/examples/bin")

add_executable(example_condensing blockSQP2/examples/example_condensing.cpp)
target_link_libraries(example_condensing blockSQP2)
set_target_properties(example_condensing
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/blockSQP2/examples/bin"
)


option(PYTHON_INTERFACE "build python interface" ON)
set(PYTHON_INTERPRETER "" CACHE STRING "The python interpreter the interface should be built for")
if(PYTHON_INTERFACE)
	if (NOT PYTHON_INTERPRETER STREQUAL "")
		message(STATUS "Using python interpreter ${PYTHON_INTERPRETER}")
		set (Python_EXECUTABLE ${PYTHON_INTERPRETER})
	endif()
	
	FetchContent_Declare(
		PYBIND11
		GIT_REPOSITORY https://github.com/pybind/pybind11.git
		GIT_TAG f5fbe867d2d26e4a0a9177a51f6e568868ad3dc8 	#v. 3.0.1
	)
	FetchContent_MakeAvailable(PYBIND11)
	
	pybind11_add_module(pyblockSQP2 "Python/pyblockSQP2.cpp")
	target_include_directories(pyblockSQP2 PUBLIC blockSQP2/include)	
	
	if(UNIX OR MINGW)
		target_compile_options(pyblockSQP2 PRIVATE 
					-Wall -Wextra -Wno-unused-parameter
					-fvisibility=hidden #Recommended in pybind11 FAQ, greatly reduces binary size
					-O3
					)
		
		#Prevent warnings for *_pointer_interface and T_array constructors 
		#that occur after compiling with -O3. Result of using int instead of size_t, should not cause any problems.
		target_link_options(pyblockSQP2 PRIVATE -Wno-alloc-size-larger-than)
	#[[
	elseif(MINGW)
		target_compile_options(pyblockSQP2 PRIVATE 
					-Wall -Wextra -Wno-unused-parameter
					-fvisibility=hidden
					-O3
					)
		target_link_options(pyblockSQP2 PRIVATE -Wno-alloc-size-larger-than)
	]]
	endif()
	
	target_link_libraries(pyblockSQP2 PUBLIC blockSQP2)

	if(MINGW)
		target_link_libraries(pyblockSQP2 PRIVATE -static)
	endif()

	set(pyblockSQP2_DIR ${CMAKE_SOURCE_DIR}/Python/blockSQP2)
	set_target_properties(pyblockSQP2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${pyblockSQP2_DIR}
												 RUNTIME_OUTPUT_DIRECTORY ${pyblockSQP2_DIR}
												 )
	copy_dynamic_libraries_to(${pyblockSQP2_DIR})
	if(MINGW)
		copy_mingw_libraries_to(${pyblockSQP2_DIR})
	endif()
endif()


#Build C interface to complete blockSQP.jl
option(JULIA_INTERFACE "build Julia interface" ON)
if(JULIA_INTERFACE)
	add_library(blockSQP2_jl SHARED C/CblockSQP2.cpp)
	target_link_libraries(blockSQP2_jl PUBLIC blockSQP2)
	
	target_include_directories(blockSQP2_jl PUBLIC blockSQP2/include)
	if(UNIX)
		target_compile_options(blockSQP2_jl PRIVATE 
					-Wall -Wextra -Wno-unused-parameter
					-fvisibility=hidden
					-O3
					)
	elseif(MINGW)
		target_compile_options(blockSQP2_jl PRIVATE 
					-Wall -Wextra -Wno-unused-parameter
					-fvisibility=hidden
					-O3
					)
		target_link_libraries(blockSQP2_jl PRIVATE -static)
	endif()
	set_target_properties(blockSQP2_jl PROPERTIES 
			PREFIX lib
			LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2.jl/bin
			RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2.jl/bin
	)
	copy_dynamic_libraries_to(${CMAKE_SOURCE_DIR}/blockSQP2.jl/bin)
endif()

