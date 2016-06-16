cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(name DemoClient)

project(${name})

add_executable(${name} ${name}.cc)


message("DEMO_INCLUDE_DIRS ${DEMO_INCLUDE_DIRS} ")
message("DEMO_LIBRARIES    ${DEMO_LIBRARIES} ")


include_directories(${DEMO_INCLUDE_DIRS})

target_link_libraries(${name} ${DEMO_LIBRARIES})


install(TARGETS ${name} DESTINATION bin)  


