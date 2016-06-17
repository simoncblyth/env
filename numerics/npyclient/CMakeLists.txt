cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NPYClient)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(OpticksBoost  REQUIRED)
find_package(GLM           REQUIRED)
find_package(BoostRap      REQUIRED)
find_package(NPY           REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}

   ${OpticksBoost_INCLUDE_DIRS}
   ${GLM_INCLUDE_DIRS}
   ${BoostRap_INCLUDE_DIRS}
   ${NPY_INCLUDE_DIRS}
)

add_definitions(
   ${OpticksBoost_DEFINITIONS}
   ${GLM_DEFINITIONS}
   ${BoostRap_DEFINITIONS}
   ${NPY_DEFINITIONS}
)

set(LIBRARIES
   ${OpticksBoost_LIBRARIES}
   ${GLM_LIBRARIES}
   ${BoostRap_LIBRARIES}
   ${NPY_LIBRARIES}
)


add_executable(${name} ${name}.cc)
target_link_libraries( ${name} ${LIBRARIES} )
install(TARGETS ${name} DESTINATION lib)  

# place .exe next to .lib and .dll for simplicity



