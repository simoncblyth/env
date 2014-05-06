cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

# TODO: avoid envvar to point to where to locate FindROOT.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{CHROMA_GEANT4_SDIR}/cmake/Modules/") 
set(name LXe)
project(${name})

find_package(Geant4 REQUIRED)
find_package(Root REQUIRED)


# TODO: replace these with FindZMQRoot.cmake
set( ZMQROOT_INCLUDE_DIR "$ENV{ZMQROOT_PREFIX}/include" )
set( ZMQROOT_LIBRARY_DIR "$ENV{ZMQROOT_PREFIX}/lib" )
set( ZMQROOT_LIBRARIES   "-lZMQRoot" )


include(${Geant4_USE_FILE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ${Geant4_INCLUDE_DIR}
                    ${ROOT_INCLUDE_DIR}
                    ${ZMQROOT_INCLUDE_DIR}
                   )


link_directories( 
#                 ${ROOT_LIBRARY_DIR} 
                 ${ZMQROOT_LIBRARY_DIR} 
                )

add_definitions( -DWITH_CHROMA_ZMQ )


ROOT_GENERATE_DICTIONARY( "ChromaPhotonListDict" 
                          "${CMAKE_CURRENT_SOURCE_DIR}/include/ChromaPhotonList.hh" 
                          LINKDEF 
                          "${CMAKE_CURRENT_SOURCE_DIR}/include/ChromaPhotonList_LinkDef.h" 
                          OPTIONS -p )

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc  )  

list( APPEND sources ChromaPhotonListDict.cxx )


add_executable(${name} EXCLUDE_FROM_ALL ${name}.cc ${sources})

target_link_libraries( ${name} 
                       ${Geant4_LIBRARIES} 
                       ${ROOT_LIBRARIES}
                       ${ZMQROOT_LIBRARIES}
                      )


# Finding ROOT headers...
#
# http://hypernews.slac.stanford.edu/HyperNews/geant4/get/installconfig/1467.html?inline=-1
# http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries
# 
# /usr/local/env/chroma_env/src/root-v5.34.14/cmake/modules/RootNewMacros.cmake
#  
#

