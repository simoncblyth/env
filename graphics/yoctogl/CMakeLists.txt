cmake_minimum_required (VERSION 3.4)
set(name ygltf_reader)
project (${name})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
 set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)


set(YBASE /usr/local/env/graphics/yoctogl/yocto-gl)

set(SOURCES
     ${YBASE}/yocto/yocto_img.cpp
     ${YBASE}/yocto/yocto_gltf.cpp
     ${name}.cc
)

include_directories(${YBASE}/yocto) 

add_executable( ${name} ${SOURCES} )

install(TARGETS ${name} DESTINATION bin)

