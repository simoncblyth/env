cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name DataModelTest)
project(${name})

list(APPEND CMAKE_MODULE_PATH "$ENV{ENV_HOME}/cmake/Modules")
include(EnvBuildOptions)

find_package(NuWaDataModel     REQUIRED)
find_package(Root              REQUIRED)
find_package(ChromaGeant4CLHEP REQUIRED)  # filching CLHEP that comes with Geant4 means cannot use standard FindCLHEP.cmake 
# NB no Geant4

include_directories(
                    ${DATAMODEL_INCLUDE_DIRS}
                    ${CLHEP_INCLUDE_DIRS}
                    ${ROOT_INCLUDE_DIR}
                   )

add_definitions( ${DATAMODEL_DEFINITIONS}  )

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cc  )  


add_executable(${name} ${sources})

target_link_libraries( ${name} 
                       ${DATAMODEL_LIBRARIES}
                       ${CLHEP_LIBRARIES}
                       ${ROOT_LIBRARIES}
                     )

install(TARGETS ${name} DESTINATION bin)

