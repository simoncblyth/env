cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
                      "$ENV{GEANT4_HOME}/cmake/Modules"   # for FindROOT.cmake 
          ) 

# need geant4-config in PATH to find G4, eg PATH=/usr/local/env/chroma_env/bin:$PATH

set(name G4DAEChromaTest)
project(${name})

find_package(G4DAEChroma REQUIRED)
find_package(ChromaPhotonList REQUIRED)
find_package(Geant4 9.5.0 REQUIRED gdml)
find_package(Root REQUIRED)
find_package(ZMQ REQUIRED)

include(EnvBuildOptions)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${Geant4_INCLUDE_DIRS}    
                    ${CHROMAPHOTONLIST_INCLUDE_DIR}
                    ${G4DAECHROMA_INCLUDE_DIRS}
                    ${ROOT_INCLUDE_DIR}
                    ${ZMQ_INCLUDE_DIR}
                   )

add_definitions(
                    ${G4DAECHROMA_DEFINITIONS}
                    ${Geant4_DEFINITIONS}    
                    ${CHROMAPHOTONLIST_DEFINITIONS}  
                    -DWITH_ZMQ
               )    


add_executable(${name} ${name}.cc )

target_link_libraries( ${name} 
                        ${Geant4_LIBRARIES} 
                        ${CHROMAPHOTONLIST_LIBRARIES}
                        ${G4DAECHROMA_LIBRARIES} 
                        ${ROOT_LIBRARIES} 
                        ${ZMQ_LIBRARIES} 
                      )

add_executable(root2npy root2npy.cc )

target_link_libraries( root2npy
                    #    ${Geant4_LIBRARIES} 
                    #    ${CHROMAPHOTONLIST_LIBRARIES}
                        ${G4DAECHROMA_LIBRARIES} 
                    #    ${ROOT_LIBRARIES} 
                    #    ${ZMQ_LIBRARIES} 
                      )



install(TARGETS ${name} DESTINATION bin)
install(TARGETS root2npy DESTINATION bin)




