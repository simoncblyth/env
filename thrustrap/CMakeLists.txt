cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)
set(name ThrustRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

# see cuda-/FindCuda.cmake
find_package(CUDA ${OPTICKS_CUDA_VERSION} REQUIRED QUIET)
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-O2;-DVERBOSE")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
find_package(PLog REQUIRED)
find_package(OpticksBoost REQUIRED)

find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(GLM  REQUIRED)
find_package(NPY  REQUIRED)
find_package(CUDARap  REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PLog_INCLUDE_DIRS} 
    ${OpticksBoost_INCLUDE_DIRS} 
    ${SysRap_INCLUDE_DIRS} 
    ${BoostRap_INCLUDE_DIRS} 
    ${GLM_INCLUDE_DIRS} 
    ${NPY_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
    ${CUDARap_INCLUDE_DIRS}
)

add_definitions(
    ${PLog_DEFINITIONS}
    ${OpticksBoost_DEFINITIONS}
    ${SysRap_DEFINITIONS}
    ${BoostRap_DEFINITIONS}
    ${CUDARap_DEFINITIONS}
)

set(LIBRARIES
    ${PLog_LIBRARIES}
    ${OpticksBoost_LIBRARIES}
    ${SysRap_LIBRARIES}
    ${BoostRap_LIBRARIES}
    ${NPY_LIBRARIES}
    ${CUDARap_LIBRARIES}
)

set(SOURCES

    THRAP_LOG.cc

    TBuf_.cu
    TBufPair_.cu
    TSparse_.cu
    TUtil_.cu
)

SET(HEADERS

    THRAP_LOG.hh
    THRAP_API_EXPORT.hh
    THRAP_HEAD.hh
    THRAP_TAIL.hh

    TBuf.hh
    TBufPair.hh
    TSparse.hh
    TUtil.hh

    strided_range.h 
    iexpand.h 
)

set(XOPT --disable-warnings)
CUDA_ADD_LIBRARY( ${name} ${SOURCES} OPTIONS -gencode=arch=compute_30,code=sm_30 ${XOPT})
target_link_libraries( ${name} ${LIBRARIES})

install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


