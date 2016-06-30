
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(name omcTest)
project(${name})

set(TEST_SOURCES
    ArgsTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ${LIBRARIES} omc)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

