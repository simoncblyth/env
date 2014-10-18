cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name MockNuWa)
project(${name})

list(APPEND CMAKE_MODULE_PATH "$ENV{GEANT4_HOME}/cmake/Modules")
find_package(Geant4 9.5.0 REQUIRED gdml)

list(APPEND CMAKE_MODULE_PATH "$ENV{ENV_HOME}/cmake/Modules")
include(EnvBuildOptions)

find_package(NuWaDataModel     REQUIRED)
find_package(Root              REQUIRED)
find_package(ChromaGeant4CLHEP REQUIRED)  # filching CLHEP that comes with Geant4 means cannot use standard FindCLHEP.cmake 

set(SRC "$ENV{LOCAL_BASE}/env/nuwa/src/DataModel")
include_directories(
                    ${SRC}/G4DataHelpers
                    ${DATAMODEL_INCLUDE_DIRS}
                    ${CLHEP_INCLUDE_DIRS}
                    ${ROOT_INCLUDE_DIR}
                    ${Geant4_INCLUDE_DIRS}  
                   )

add_definitions( ${DATAMODEL_DEFINITIONS}  )

file(GLOB sources 
                   ${SRC}/G4DataHelpers/src/lib/*.cc
                   ${CMAKE_CURRENT_SOURCE_DIR}/*.cc  
    )  


add_executable(${name} ${sources})

target_link_libraries( ${name} 
                       ${DATAMODEL_LIBRARIES}
                       ${CLHEP_LIBRARIES}
                       ${ROOT_LIBRARIES}
                       ${Geant4_LIBRARIES} 
                     )

install(TARGETS ${name} DESTINATION bin)

