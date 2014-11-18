cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name testrapsqlite)
project(${name})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-to-void-pointer-cast -Wreorder ")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Wno-int-to-void-pointer-cast")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
   )


find_package(RapSqlite REQUIRED)

include(EnvBuildOptions)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RapSqlite_INCLUDE_DIRS}    
)


message( STATUS "RapSqlite_INCLUDE_DIRS ${RapSqlite_INCLUDE_DIRS} ")
message( STATUS "RapSqlite_LIBRARIES    ${RapSqlite_LIBRARIES} ")

add_executable(${name} ${name}.cc )

target_link_libraries( ${name} 
                        ${RapSqlite_LIBRARIES} 
                      )


install(TARGETS ${name} DESTINATION bin)



