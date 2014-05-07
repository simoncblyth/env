cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

# TODO: avoid envvar to point to where to locate FindROOT.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{CHROMA_GEANT4_SDIR}/cmake/Modules" 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


set(name LXe)
project(${name})

find_package(Geant4 REQUIRED)
find_package(Root REQUIRED)
find_package(ZMQRoot REQUIRED)
find_package(ChromaPhotonList REQUIRED)

include(EnvBuildOptions)
include(${Geant4_USE_FILE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ${Geant4_INCLUDE_DIR}
                    ${ROOT_INCLUDE_DIR}
                    ${ZMQROOT_INCLUDE_DIR}
                    ${CHROMAPHOTONLIST_INCLUDE_DIR}
                   )

add_definitions( -DWITH_CHROMA_ZMQ )

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc  )  

add_executable(${name} EXCLUDE_FROM_ALL ${name}.cc ${sources})

target_link_libraries( ${name} 
                       ${Geant4_LIBRARIES} 
                       ${ROOT_LIBRARIES}
                       ${ZMQROOT_LIBRARIES}
                       ${CHROMAPHOTONLIST_LIBRARIES}
                      )

