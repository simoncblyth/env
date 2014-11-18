cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name testcjson)
project(${name})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-to-void-pointer-cast -Wreorder ")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Wno-int-to-void-pointer-cast")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
   )


find_package(cJSON REQUIRED)

include(EnvBuildOptions)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${cJSON_INCLUDE_DIRS}    
)


message( STATUS "cJSON_INCLUDE_DIRS ${cJSON_INCLUDE_DIRS} ")
message( STATUS "cJSON_LIBRARIES    ${cJSON_LIBRARIES} ")

add_executable(${name} ${name}.c )

target_link_libraries( ${name} 
                        ${cJSON_LIBRARIES} 
                      )

install(TARGETS ${name} DESTINATION bin)

