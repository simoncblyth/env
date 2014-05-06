cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{CHROMA_GEANT4_SDIR}/cmake/Modules/")  # to access FindROOT.cmake
set(name LXe)
project(${name})

find_package(Geant4 REQUIRED)
find_package(Root REQUIRED)


set( ZMQ_INCLUDE_DIR "$ENV{ZEROMQ_PREFIX}/include" )
set( ZMQ_LIBRARY_DIR "$ENV{ZEROMQ_PREFIX}/lib" )
set( ZMQ_LIBRARIES   "-lzmq" )


include(${Geant4_USE_FILE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ${Geant4_INCLUDE_DIR}
                    ${ROOT_INCLUDE_DIR}
                    ${ZMQ_INCLUDE_DIR}
                   )

link_directories( 
                 ${ROOT_LIBRARY_DIR} 
                 ${ZMQ_LIBRARY_DIR} 
                )

add_definitions( -DWITH_CHROMA_ZMQ )


ROOT_GENERATE_DICTIONARY( "${CMAKE_CURRENT_SOURCE_DIR}/src/ChromaPhotonListDict" 
                          "${CMAKE_CURRENT_SOURCE_DIR}/include/ChromaPhotonList.hh" 
                          LINKDEF 
                          "${CMAKE_CURRENT_SOURCE_DIR}/include/ChromaPhotonList_LinkDef.h" 
                          OPTIONS -p )

ROOT_GENERATE_DICTIONARY( "${CMAKE_CURRENT_SOURCE_DIR}/src/MyTMessageDict"  
                          "${CMAKE_CURRENT_SOURCE_DIR}/include/MyTMessage.hh"      
                          LINKDEF 
                          "${CMAKE_CURRENT_SOURCE_DIR}/include/MyTMessage_LinkDef.h"       
                          OPTIONS -p )


file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc  )   # sets the variable ${sources} to the list of matched files

list( APPEND sources src/ChromaPhotonListDict.cxx src/MyTMessageDict.cxx )





add_executable(${name} EXCLUDE_FROM_ALL ${name}.cc ${sources})

target_link_libraries( ${name} 
                       ${Geant4_LIBRARIES} 
                       ${ROOT_LIBRARIES}
                       ${ZMQ_LIBRARIES}
                      )


# Finding ROOT headers...
#
# http://hypernews.slac.stanford.edu/HyperNews/geant4/get/installconfig/1467.html?inline=-1
# http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries
# 
# /usr/local/env/chroma_env/src/root-v5.34.14/cmake/modules/RootNewMacros.cmake
#  
#

