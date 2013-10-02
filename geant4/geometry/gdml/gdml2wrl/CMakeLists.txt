
# http://geant4.cern.ch/support/source/geant4/cmake/Modules/FindXercesC.cmake
# http://www.cmake.org/cmake/help/cmake_tutorial.html

cmake_minimum_required (VERSION 2.6)
project (gdml2wrl)

include_directories (${NUWA_G4_INCDIR}) 
include_directories (${NUWA_CLHEP_INCDIR}) 
include_directories (${NUWA_XERCESC_INCDIR}) 

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

add_executable(gdml2wrl gdml2wrl.cxx)


