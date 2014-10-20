cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
                      "$ENV{GEANT4_HOME}/cmake/Modules"   # for FindROOT.cmake 
          ) 

# need geant4-config in PATH to find G4, eg PATH=/usr/local/env/chroma_env/bin:$PATH

set(name G4DAEChromaTest)
project(${name})

find_package(G4DAEChroma REQUIRED)
find_package(Geant4 9.5.0 REQUIRED gdml)

include(EnvBuildOptions)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${Geant4_INCLUDE_DIRS}    
                    ${G4DAECHROMA_INCLUDE_DIRS}
                   )

add_definitions(
                    ${G4DAECHROMA_DEFINITIONS}
                    ${Geant4_DEFINITIONS}    
               )    


add_executable(${name} ${name}.cc )

target_link_libraries( ${name} 
                       ${Geant4_LIBRARIES} 
                       ${G4DAECHROMA_LIBRARIES} 
                      )

install(TARGETS ${name} DESTINATION bin)


