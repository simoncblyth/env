cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name cnpytest)
project(${name})

set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH}  
    "$ENV{ENV_HOME}/cmake/Modules"
    "$ENV{GEANT4_HOME}/cmake/Modules" 
)

include(EnvBuildOptions)

find_package(CNPY REQUIRED)
find_package(Geant4 9.5.0 REQUIRED gdml) 

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${Geant4_INCLUDE_DIRS}
    ${CNPY_INCLUDE_DIRS}
)

add_definitions(
    ${Geant4_DEFINITIONS}    
)    

add_executable(${name} ${name}.cc)

target_link_libraries( ${name} 
    ${CNPY_LIBRARIES}
    ${Geant4_LIBRARIES} 
)

install(TARGETS ${name} DESTINATION bin)

