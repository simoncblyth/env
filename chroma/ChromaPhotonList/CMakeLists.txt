cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{GEANT4_HOME}/cmake/Modules" 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(name ChromaPhotonList)
project(${name})

find_package(Geant4 REQUIRED)
find_package(Root REQUIRED)

include(EnvBuildOptions)
include(${Geant4_USE_FILE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${Geant4_INCLUDE_DIR}
                    ${ROOT_INCLUDE_DIR}
                   )

ROOT_GENERATE_DICTIONARY( "ChromaPhotonListDict" 
                          "${CMAKE_CURRENT_SOURCE_DIR}/Chroma/ChromaPhotonList.hh" 
                          LINKDEF 
                          "${CMAKE_CURRENT_SOURCE_DIR}/dict/ChromaPhotonList_LinkDef.h" 
                          OPTIONS -p )

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc  )  
list(APPEND sources ChromaPhotonListDict.cxx )

add_definitions(
                   ${Geant4_DEFINITIONS} 
                  -DCPL_WITH_GEANT4  
                )    



# not easy to move header to subdir due to 

add_library(${name} SHARED ${sources})

target_link_libraries( ${name} 
                       ${Geant4_LIBRARIES} 
                       ${ROOT_LIBRARIES}
                      )

install(FILES "Chroma/ChromaPhotonList.hh" DESTINATION include/Chroma)
install(TARGETS ${name} LIBRARY DESTINATION lib)




