cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name RapSqlite)
project(${name})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-to-void-pointer-cast -Wreorder ")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Wno-int-to-void-pointer-cast")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
   )


find_package(SQLite3 REQUIRED)

include(EnvBuildOptions)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SQLite3_INCLUDE_DIRS}    
)

add_definitions(
)

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc  
)  

add_library(${name} SHARED ${sources})
target_link_libraries( ${name} 
    ${SQLite3_LIBRARIES} 
)


install(TARGETS ${name} LIBRARY DESTINATION lib)

# arranging header installation to retain containing folder structure
file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/*/*.hh)
foreach(abspath ${headers})
   string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" hdr "${abspath}")  # remove the SRC prefix 
   get_filename_component(hbase ${hdr} NAME)
   get_filename_component(hpath ${hdr} PATH)
   get_filename_component(hfold ${hpath} NAME)
   #message( DEBUG " hdr ${hdr} hbase ${hbase} hpath ${hpath} hfold ${hfold}")
   install(FILES ${abspath} DESTINATION include/${hfold})
endforeach(abspath)


