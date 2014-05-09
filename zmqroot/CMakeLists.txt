cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
#
# Currently depends on envvars::
#
#    chroma-
#    chroma-geant4-export 
#        CHROMA_GEANT4_SDIR=/usr/local/env/chroma_env/src/geant4.9.5.p01
#
#    zeromq-
#

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{CHROMA_GEANT4_SDIR}/cmake/Modules"   # for FindROOT.cmake
                      "$ENV{ENV_HOME}/cmake/Modules"             # for FindZMQ.cmake 
                   )  


set(name ZMQRoot)
project(${name})

find_package(Root REQUIRED)
find_package(ZMQ REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ${ROOT_INCLUDE_DIR}
                    ${ZMQ_INCLUDE_DIR}
                   )


include(EnvBuildOptions)   # RPATH settings


add_definitions( "-DWITH_ZMQ" )

# do not "link_directories" more that necessary,
# with ROOT_LIBRARY_DIR present results in a cmake rpath bug, 
# causing install_name_tool corruption at install, see `cmake-`
#
#link_directories( 
#                 ${ZMQ_LIBRARY_DIR} 
#                )


# Generates MyTMessageDict.cxx, MyTMessageDict.h into build dir CMAKE_CURRENT_BINARY_DIR  
ROOT_GENERATE_DICTIONARY( "MyTMessageDict"  
                          "${CMAKE_CURRENT_SOURCE_DIR}/include/MyTMessage.hh"      
                          LINKDEF 
                          "${CMAKE_CURRENT_SOURCE_DIR}/include/MyTMessage_LinkDef.h"       
                          OPTIONS -p )


#  needs gccxml
#REFLEX_GENERATE_DICTIONARY( "MyTMessageDict"
#                            "${CMAKE_CURRENT_SOURCE_DIR}/include/MyTMessage.hh"      
#                            SELECTION
#                            "${CMAKE_CURRENT_SOURCE_DIR}/dict/classes.xml"      
#                         )



file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc  ) 
list(APPEND sources MyTMessageDict.cxx )  


add_library(${name} SHARED ${sources})

target_link_libraries( ${name} 
                       ${ROOT_LIBRARIES}
                       ${ZMQ_LIBRARIES}
                      )


install(FILES "include/ZMQRoot.hh" DESTINATION include)
install(TARGETS ${name} LIBRARY DESTINATION lib)



