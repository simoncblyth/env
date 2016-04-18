cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OVRMINIMAL)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(OpenVR REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${OpenVR_INCLUDE_DIRS} 
)

add_definitions( 
   ${GLEW_DEFINITIONS} 
   ${GLFW_DEFINITIONS} 
   ${OpenVR_DEFINITIONS} 
)
#   -D_VR 

add_executable( ${name}  
    main.cpp
)

set(LIBRARIES
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OpenVR_LIBRARIES}
)

target_link_libraries( ${name} ${LIBRARIES} )

install(TARGETS ${name}  DESTINATION bin)


