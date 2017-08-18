cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name TXF)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{OPTICKS_HOME}/cmake/Modules"
          ) 


include(EnvCompilationFlags) 
include(EnvBuildOptions)

message("OPTICKS_PREFIX:${OPTICKS_PREFIX}")

########### using env/cmake/Modules/*.cmake #####

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLM    REQUIRED)
find_library(OpenGL_LIBRARY NAMES GL )


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
)

set(DEFINITIONS
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${GLM_DEFINITIONS}
)

set(LIBRARIES
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${GLM_LIBRARIES}
      ${OpenGL_LIBRARY}
)

include_directories( ${INCLUDE_DIRS} )

add_definitions( ${DEFINITIONS} )

add_executable( ${name} txf.cc ) 

target_link_libraries( ${name} ${LIBRARIES})

install(TARGETS ${name} DESTINATION lib)

#add_subdirectory(tests)

