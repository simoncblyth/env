cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name ThrustOpenGLInterop)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{OPTICKS_HOME}/cmake/Modules") 

include(EnvBuildOptions)


########### using env/cmake/Modules/*.cmake #####

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_library(OpenGL_LIBRARY NAMES GL )
find_package(CUDA REQUIRED)


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${CUDA_INCLUDE_DIRS} 
)
set(DEFINITIONS
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${CUDA_DEFINITIONS}
)
set(LIBRARIES
   ${GLEW_LIBRARIES}
   ${GLFW_LIBRARIES}
   ${OpenGL_LIBRARY}
   ${CUDA_LIBRARIES}
)


include_directories( ${INCLUDE_DIRS} )

add_definitions( ${DEFINITIONS} )

cuda_add_executable( ${name} thrust_opengl_interop.cu ) 

target_link_libraries( ${name} ${LIBRARIES})

install(TARGETS ${name} DESTINATION bin)

#add_subdirectory(tests)

