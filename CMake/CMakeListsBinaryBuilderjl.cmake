cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Please use out of source build:
  cmake -B build
  {make -C build } OR {cmake --build build}")
endif()

project(BinaryBuilderblockSQP2
	DESCRIPTION "Build system for blockSQP2 to be used in BinaryBuilder.jl."
	HOMEPAGE_URL https://github.com/ReWittmann/blockSQP2
    LANGUAGES CXX
	)
set(CMAKE_CXX_STANDARD 20)



# find_path(INCLUDE_PREFIX
# 	include/dmumps_c.h
# 	REQUIRED
# )
# message(STATUS "Found include prefix: " ${INCLUDE_PREFIX})

# find_library(MUMPS_LIBRARY
# 	"dmumps"
# 	REQUIRED
# )
# message(STATUS "Found mumps library:" ${MUMPS_LIBRARY})

# find_library(MUMPS_COMMON_LIBRARY
# 	"mumps_common"
# 	REQUIRED
# )
# message(STATUS "Found mumps common library:" ${MUMPS_COMMON_LIBRARY})

# find_library(MUMPS_PORD_LIBRARY
# 	"pord"
# 	REQUIRED
# )
# message(STATUS "Found pord library:" ${MUMPS_PORD_LIBRARY})

# find_library(MUMPS_MPISEQ_LIBRARY
# 	"mpiseq"
# 	REQUIRED
# )
# message(STATUS "Found mumps mpiseq library:" ${MUMPS_MPISEQ_LIBRARY})
message(STATUS "Building for " ${CMAKE_SYSTEM_NAME})

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	message(STATUS "Searching for libblastrampoline-5.dll ...")
	set(BLASTRAMPOLINE_LIBNAME blastrampoline-5)
else()
	set(BLASTRAMPOLINE_LIBNAME blastrampoline)
endif()

find_library(LIBBLASTRAMPOLINE_LIBRARY
	${BLASTRAMPOLINE_LIBNAME}
	REQUIRED
)
message(STATUS "Found LIBBLASTRAMPOLINE_LIBRARY: " ${LIBBLASTRAMPOLINE_LIBRARY})


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
target_compile_definitions(qpOASES PUBLIC SOLVER_MUMPS PRIVATE __NO_COPYRIGHT__
	QPOS_CBLAS_SUFFIX=${CBLAS_SUFFIX}
)

IF (UNIX)
	target_compile_definitions(qpOASES PRIVATE LINUX)
	target_compile_options(qpOASES PUBLIC -fPIC PRIVATE -O3 INTERFACE -Wno-overloaded-virtual)
	# if (NOT APPLE)
	# 	target_link_libraries(qpOASES PUBLIC -lgfortran)
	# endif()
ELSEIF (WIN32)
    #target_compile_options(qpOASES PRIVATE -nologo -EHsc -DWIN32)
	if(MINGW)
		target_compile_options(qpOASES PRIVATE -DWIN32 
			-Wall -pedantic -Wfloat-equal -Wshadow 
			-O3 -finline-functions
		)
	endif()
ENDIF ()

target_include_directories(qpOASES
	PUBLIC	${qpOASES_MODDED_DIR}/include ${qpOASES_MODDED_DIR}/include/qpOASES 
	PUBLIC ${INCLUDE_PREFIX}/include
)

# set(QPOASES_LIBRARIES qpOASES ${MUMPS_LIBARY} ${MUMPS_COMMON_LIBRARY} ${MUMPS_PORD_LIBRARY} ${MUMPS_MPISEQ_LIBRARY} ${LIBBLASTRAMPOLINE_LIBRARY})

set(QPOASES_LIBRARIES qpOASES 
    ${DMUMPS_LIBRARIES} \
    ${DMUMPS_COMMON_LIBRARY} \
    ${DMUMPS_PORD_LIBRARY} \
    ${DMUMPS_MPISEQ_LIBRARY} \
	${LIBBLASTRAMPOLINE_LIBRARY})

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
target_compile_definitions(blockSQP2 PRIVATE QPSOLVER_QPOASES SOLVER_MUMPS BSQP_PAR_QPS_DISABLED 
						BSQP_CBLAS_SUFFIX=${CBLAS_SUFFIX})

if (UNIX AND NOT APPLE)
	target_compile_definitions(blockSQP2 PUBLIC LINUX)
elseif(WIN32)
	target_compile_definitions(blockSQP2 PUBLIC WINDOWS)
endif()
target_compile_options(blockSQP2 PRIVATE 
	-Wall -Wextra -Wno-unused-parameter -Wno-maybe-uninitialized -fPIC
	)

set_target_properties(blockSQP2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/bin
										   ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/lib
										   RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2/bin
										   )
target_include_directories(blockSQP2 PUBLIC ${CMAKE_SOURCE_DIR}/blockSQP2/include)
target_link_libraries(blockSQP2 PUBLIC ${QPOASES_LIBRARIES})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

#Build C interface to complete blockSQP.jl
add_library(blockSQP2_jl SHARED ${CMAKE_SOURCE_DIR}/C/CblockSQP2.cpp)
target_link_libraries(blockSQP2_jl PUBLIC blockSQP2)

target_compile_options(blockSQP2_jl PRIVATE 
			-Wall -Wextra -Wno-unused-parameter
			-fvisibility=hidden
			-fPIC
			)
set_target_properties(blockSQP2_jl PROPERTIES 
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2.jl/bin
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/blockSQP2.jl/bin
)

install(TARGETS blockSQP2_jl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})