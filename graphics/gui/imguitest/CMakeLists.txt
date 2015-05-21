cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name ImGuiTest)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)

set(BUILD_SHARED_LIBS ON)

## c++11 forced by AsioZMQ   needs to come after OptiX
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif ()


########### using env/cmake/Modules/*.cmake #####

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)

find_package(ImGui REQUIRED)



include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 

   ${ImGui_INCLUDE_DIRS}
)

add_definitions(
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${ImGui_DEFINITIONS}
)


message("ImGui_LIBRARIES:" ${ImGui_LIBRARIES})  
message("ImGui_INCLUDE_DIRS:" ${ImGui_INCLUDE_DIRS})  

add_executable( ${name} main.cpp )
target_link_libraries( ${name} 
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${ImGui_LIBRARIES})

install(TARGETS ${name} DESTINATION bin)



