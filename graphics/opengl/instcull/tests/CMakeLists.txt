cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name INSTCULLTest)
project(${name})

set(TEST_SOURCES
    instcull0.cc
    instcull1.cc
    instance0.cc
    instance.cc
    CompTest.cc
)
 
foreach(TEST_SOURCE ${TEST_SOURCES})

    get_filename_component(TGT ${TEST_SOURCE} NAME_WE)

    add_executable(${TGT} ${TEST_SOURCE})

    target_link_libraries(${TGT} ${LIBRARIES} INSTCULL)

    install(TARGETS ${TGT} DESTINATION lib)

endforeach()

