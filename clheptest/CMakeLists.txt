cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(name CLHEPTest)

project(${name})

list(APPEND CMAKE_MODULE_PATH "$ENV{ENV_HOME}/cmake/Modules")

include(EnvBuildOptions) 

find_package(CLHEP REQUIRED)





