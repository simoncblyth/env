cmake_minimum_required(VERSION 3.15...3.27)

set(name my_project)
project(${name})

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# https://nanobind.readthedocs.io/en/latest/building.html#building

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

message( STATUS "${name} nanobind_ROOT : ${nanobind_ROOT} ")

find_package(nanobind CONFIG REQUIRED)

message(STATUS "${name} nanobind_FOUND     : ${nanobind_FOUND} ")

# https://github.com/search?q=nanobind_add_module&type=code
# HOW TO INSTALL, FIND THE NAME ...

nanobind_add_module(my_ext
   my_ext.cpp
   add.cpp
   dog.cpp
)





