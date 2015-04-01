cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(name NPYAsioZMQ)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)
#include(EnvBuildOptions)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif ()


find_package(Boost REQUIRED COMPONENTS system)
find_package(ZMQ REQUIRED)
find_package(AsioZMQ REQUIRED)


message(ZMQ_LIBRARIES:${ZMQ_LIBRARIES})
message(ZMQ_LIBRARY:${ZMQ_LIBRARY})
message(Boost_LIBRARIES:${Boost_LIBRARIES})


include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${ZMQ_INCLUDE_DIRS} 
   ${AsioZMQ_INCLUDE_DIRS} 
)

add_definitions(
   ${Boost_DEFINITIONS}
   ${ZMQ_DEFINITIONS}
   ${AsioZMQ_DEFINITIONS}
)

add_library( ${name} SHARED
          npyworker.cc 
          npyQueue.cc 
          npyBuffer.cc 
)

target_link_libraries( ${name}
    ${Boost_LIBRARIES}
    ${ZMQ_LIBRARIES} 
)


add_executable(        ${name}Test npyasiozmqtest.cc )

target_link_libraries( ${name}Test 
    ${Boost_LIBRARIES}
    ${ZMQ_LIBRARIES} 
    ${name}
)



