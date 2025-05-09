cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name INSTCULL)
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
find_package(GLEQ REQUIRED)
find_package(GLM    REQUIRED)
find_library(OpenGL_LIBRARY NAMES GL )


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${GLEQ_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
)

set(DEFINITIONS
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${GLEQ_DEFINITIONS}
   ${GLM_DEFINITIONS}
)

set(LIBRARIES
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${GLEQ_LIBRARIES}
      ${GLM_LIBRARIES}
      ${OpenGL_LIBRARY}
)

set(SOURCES
     Demo.cc
     ICDemo.cc
     Prog.cc
     Frame.cc
     Buf.cc
     Buf4.cc
     Att.cc
     Shader.cc
     Tra.cc
     Cam.cc
     Vue.cc
     Comp.cc
     Geom.cc
     G.cc
     GU.cc
     BB.cc
     Pos.cc
     UV.cc
     Box.cc
     Tri.cc
     Prim.cc
     Cube.cc
     Sphere.cc
     InstShader.cc
     CullShader.cc
     LODCullShader.cc
     SContext.cc

)
 

include_directories( ${INCLUDE_DIRS} )
add_definitions( ${DEFINITIONS} )
add_library( ${name}  SHARED ${SOURCES} )
target_link_libraries( ${name} ${LIBRARIES})
install(TARGETS ${name} DESTINATION lib)

add_subdirectory(tests)

