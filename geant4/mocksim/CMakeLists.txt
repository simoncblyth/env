cmake_minimum_required(VERSION 2.8)
find_package(Geant4 9.5.0 REQUIRED gdml)

include_directories(${Geant4_INCLUDE_DIRS})       
add_definitions(${Geant4_DEFINITIONS})            
set(CMAKE_CXX_FLAGS ${Geant4_CXX_FLAGS})            

add_executable(mocksim mocksim.cc)                  
target_link_libraries(mocksim ${Geant4_LIBRARIES})  




