cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name CUDARapTest)
project(${name})

#message("${name}.INCLUDE_DIRS : ${INCLUDE_DIRS}")
#message("${name}.LIBRARIES    : ${LIBRARIES}")

set(TEST_SOURCES
   LaunchSequenceTest.cc
   cuRANDWrapperTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})
    target_link_libraries(${TGT} ${LIBRARIES} )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()




