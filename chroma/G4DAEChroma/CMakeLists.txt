cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name G4DAEChroma)
project(${name})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-to-void-pointer-cast -Wreorder ")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Wno-int-to-void-pointer-cast")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
                      "$ENV{GEANT4_HOME}/cmake/Modules"   # for FindROOT.cmake
          ) 

message( STATUS "CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ")

find_package(Geant4 9.5.0 REQUIRED gdml)
find_package(CNPY REQUIRED)
find_package(ChromaPhotonList REQUIRED)
find_package(ZMQRoot REQUIRED)
find_package(Root REQUIRED)

include(EnvBuildOptions)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Geant4_INCLUDE_DIRS}    
    ${CHROMAPHOTONLIST_INCLUDE_DIR}
    ${ZMQROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CNPY_INCLUDE_DIRS}
)

add_definitions(
    ${Geant4_DEFINITIONS} 
    -DWITH_CHROMA_ZMQ  -DEXPORT_G4GDML
    ${CHROMAPHOTONLIST_DEFINITIONS}  
)    

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc  
)  

add_library(${name} SHARED ${sources})
target_link_libraries( ${name} 
    ${Geant4_LIBRARIES} 
    ${CHROMAPHOTONLIST_LIBRARIES}
    ${ZMQROOT_LIBRARIES}
    ${ROOT_LIBRARIES}
    ${CNPY_LIBRARIES}
)

install(TARGETS ${name} LIBRARY DESTINATION lib)



# arranging header installation to retain containing folder structure
file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/*/*.hh)
foreach(abspath ${headers})
   string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" hdr "${abspath}")  # remove the SRC prefix 
   get_filename_component(hbase ${hdr} NAME) 
   get_filename_component(hpath ${hdr} PATH) 
   get_filename_component(hfold ${hpath} NAME) 
   #message( DEBUG " hdr ${hdr} hbase ${hbase} hpath ${hpath} hfold ${hfold}")
   install(FILES ${abspath} DESTINATION include/${hfold})
endforeach(abspath)



