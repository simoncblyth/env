cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{GEANT4_HOME}/cmake/Modules"    # for FindROOT
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


set(name DataModel)
project(${name})

set(DMD "/usr/local/env/nuwa/DataModel")

find_package(Root REQUIRED)

include_directories(
                    ${DMD}/Context
                    ${DMD}/BaseEvent
                    ${DMD}/Conventions
                    ${DMD}/SimEvent
                    ${DMD}/GaudiKernel
                    ${DMD}/DybKernel
                    /usr/local/env/chroma_env/include/Geant4  # contains CLHEP
                    ${ROOT_INCLUDE_DIR}
                   )

file(GLOB sources 
            ${DMD}/*/src/*.cc 
            ${DMD}/GaudiKernel/src/Lib/*.cpp
       )  

file(GLOB headers "${DMD}/*/*/*.h")

add_definitions( -DGOD_NOALLOC  )

add_library(${name} SHARED ${sources})

target_link_libraries( ${name} 
                       /usr/local/env/chroma_env/lib/libG4clhep.dylib
                       ${ROOT_LIBRARIES}
                      )

# arranging header installation to retain containing folder structure
foreach(abspath ${headers})
   string(REPLACE "${DMD}/" "" hdr "${abspath}")
   get_filename_component(hbase ${hdr} NAME) 
   get_filename_component(hpath ${hdr} PATH) 
   get_filename_component(hfold ${hpath} NAME) 
   message( STATUS "hdr ${hdr} hbase ${hbase} hpath ${hpath} hfold ${hfold}")
   install(FILES ${abspath} DESTINATION include/${hfold})
endforeach(hdr)

install(TARGETS ${name} LIBRARY DESTINATION lib)




