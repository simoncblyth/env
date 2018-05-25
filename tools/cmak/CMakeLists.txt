cmake_minimum_required(VERSION 3.11)

set(CMAKE_PREFIX_PATH "/prefix") 
set(as_subproject Foo)

macro(find_package)
  if(NOT "${ARG0}" IN_LIST as_subproject)
    _find_package(${ARGV})
  endif()
endmacro()

add_subdirectory(Foo)
add_subdirectory(App)


