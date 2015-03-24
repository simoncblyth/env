cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name VLTest)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
                      "$ENV{LOCAL_BASE}/env/graphics/vl/cmake"
          ) 


set(VL_ROOT $ENV{LOCAL_BASE}/env/graphics/vl)

find_package(GLFW REQUIRED)

# for static linking do this
# find_package(VL COMPONENTS VLCore VLGraphics VLVG VLVolume VLMolecule VLWin32 FreeType JPG PNG TIFF ZLib REQUIRED)
# add_definitions(-DVL_STATIC_LINKING)
# - FreeType, JPG, PNG, TIFF, ZLib, are not required with dynamic linking because they have been linked already in VLCore and VLGraphics.

find_package(VL COMPONENTS VLCore VLGraphics VLVG VLVolume VLMolecule REQUIRED)


include(EnvBuildOptions)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLFW_INCLUDE_DIRS} 
   ${VL_INCLUDE_DIRS}
)

add_definitions(
   ${GLFW_DEFINITIONS}
)


add_executable( ${name} vltest.cc )

target_link_libraries( ${name} 
     ${GLFW_LIBRARIES}
     ${VL_LIBRARIES}
)

install(TARGETS ${name} DESTINATION bin)



