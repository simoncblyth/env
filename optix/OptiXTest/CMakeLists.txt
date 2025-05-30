cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(name OptiXTest)

project(${name})



# Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
# is a custom extension to the FindCUDA code distributed by CMake.
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{OPTIX_SDK_DIR}/CMake"
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


find_package(AssimpWrap REQUIRED)

find_package(OptiX REQUIRED)

find_package(CUDA 2.3 REQUIRED)



include(FindSUtilGLUT)

include(EnvBuildOptions)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OptiX_INCLUDE}  
    $ENV{OPTIX_SDK_DIR}/sutil
    ${CUDA_INCLUDE_DIRS}
)

add_definitions(
    ${OptiX_DEFINITIONS} 
)


function(add_optix_executable target_name)

    # split arguments into three lists 
    CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

    # Create the rules to build the PTX from the CUDA files.
    CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options} )

    add_executable(${target_name}
        ${source_files}
        ${generated_files}
        ${cmake_options}
    )


    find_library( SUTIL_LIBRARY
              NAMES sutil
              PATHS $ENV{OPTIX_SAMPLES_INSTALL_DIR}/lib )

    message("SUTIL_LIBRARY ${SUTIL_LIBRARY} ")


    target_link_libraries( ${target_name} 
                           ${SUTIL_LIBRARY}
                           optix
                         )

endfunction()



add_optix_executable(${name} 
                     ${name}.cc 
                     draw_color.cu
)

# why have to manually find sutil, but not optix ?
#install(TARGETS ${name} DESTINATION bin)


