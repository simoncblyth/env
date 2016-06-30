cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(name omc)
project(${name})


set(CMAKE_MODULE_PATH "$ENV{ENV_HOME}/cmake/Modules")

include(EnvBuildOptions)  # RPATH setup
include(CTest)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})


set(pkg OpenMesh)
find_package(${pkg})

message("${pkg}_LIBRARY       : ${${pkg}_LIBRARY}")
message("${pkg}_LIBRARIES     : ${${pkg}_LIBRARIES}")
message("${pkg}_INCLUDE_DIRS  : ${${pkg}_INCLUDE_DIRS}")
message("${pkg}_DEFINITIONS   : ${${pkg}_DEFINITIONS}")

set(LIBRARIES
   ${${pkg}_LIBRARIES}
)

message("${name}:LIBRARIES : ${LIBRARIES} ")

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}
   ${${pkg}_INCLUDE_DIRS} 
)

add_definitions(
    ${${pkg}_DEFINITIONS} 
)

set(SOURCES
    Args.cc
) 

set(HEADERS
    Args.hh
)

add_library( ${name}  SHARED ${SOURCES})
target_link_libraries( ${name}  ${LIBRARIES} )

install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


