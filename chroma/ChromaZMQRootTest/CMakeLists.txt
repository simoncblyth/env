cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{GEANT4_HOME}/cmake/Modules"   # for FindROOT.cmake
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(name ChromaZMQRootTest)
project(${name})

find_package(ChromaPhotonList REQUIRED)
find_package(ZMQRoot REQUIRED)
find_package(Root REQUIRED)

include(EnvBuildOptions)

#add_definitions( "-DWITH_ZMQ" )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CHROMAPHOTONLIST_INCLUDE_DIR}
                    ${ZMQROOT_INCLUDE_DIR}
                    ${ROOT_INCLUDE_DIR}
                   )

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cc  )  

add_executable(${name} ${sources})

target_link_libraries( ${name} 
                       ${CHROMAPHOTONLIST_LIBRARIES}
                       ${ZMQROOT_LIBRARIES}
                       ${ROOT_LIBRARIES}
                      )

install(TARGETS ${name} DESTINATION bin)

