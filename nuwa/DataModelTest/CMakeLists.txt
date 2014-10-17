cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{GEANT4_HOME}/cmake/Modules"    # for FindROOT
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


message( STATUS "CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

set(name DataModelTest)
project(${name})

set(DATAMODEL_PREFIX "/usr/local/env/nuwa")

find_package(Geant4 REQUIRED)  # bizarre, not using G4 yet : this is just to use its FindROOT
find_package(Root REQUIRED)

include_directories(
                    ${DATAMODEL_PREFIX}/include
                    /usr/local/env/chroma_env/include/Geant4  # contains CLHEP dir
                    ${ROOT_INCLUDE_DIR}
                   )


add_definitions( -DGOD_NOALLOC  )

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cc  )  

add_executable(${name} ${sources})

target_link_libraries( ${name} 
                     -L${DATAMODEL_PREFIX}/lib -lDataModel 
                        /usr/local/env/chroma_env/lib/libG4clhep.dylib 
                       ${ROOT_LIBRARIES}
                      )


# huh rpath not being done ? have to set DYLD_...
install(TARGETS ${name} DESTINATION bin)

