cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name DataModel)
project(${name})

list(APPEND CMAKE_MODULE_PATH "$ENV{ENV_HOME}/cmake/Modules")
include(EnvBuildOptions)   # RPATH settings
find_package(Root REQUIRED)
find_package(ChromaGeant4CLHEP REQUIRED)

#message( STATUS "CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

set(SRC "$ENV{LOCAL_BASE}/env/nuwa/src/DataModel")
include_directories(
                    ${SRC}/Context
                    ${SRC}/BaseEvent
                    ${SRC}/Conventions
                    ${SRC}/SimEvent
                    ${SRC}/GaudiKernel
                    ${SRC}/DybKernel
                    ${SRC}/G4DataHelpers
                    ${CLHEP_INCLUDE_DIRS}
                    ${ROOT_INCLUDE_DIR}
                   )

file(GLOB headers  ${SRC}/*/*/*.h)
file(GLOB sources 
                   ${SRC}/*/src/*.cc 
                   ${SRC}/GaudiKernel/src/Lib/*.cpp
       )  

add_definitions( -DGOD_NOALLOC  )

add_library(${name} SHARED ${sources})
target_link_libraries( ${name} 
                       ${CLHEP_LIBRARIES}
                       ${ROOT_LIBRARIES}
                      )

# arranging header installation to retain containing folder structure
foreach(abspath ${headers})
   string(REPLACE "${SRC}/" "" hdr "${abspath}")  # remove the SRC prefix 
   get_filename_component(hbase ${hdr} NAME) 
   get_filename_component(hpath ${hdr} PATH) 
   get_filename_component(hfold ${hpath} NAME) 
   #message( DEBUG "hdr ${hdr} hbase ${hbase} hpath ${hpath} hfold ${hfold}")
   install(FILES ${abspath} DESTINATION include/${hfold})
endforeach(abspath)

install(TARGETS ${name} LIBRARY DESTINATION lib)


