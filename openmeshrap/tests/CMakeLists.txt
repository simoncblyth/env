cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OpenMeshRapTest)
project(${name})

#file(GLOB TEST_SOURCES "*Test.cc")
set(TEST_SOURCES

    DeleteFaceTest.cc
)



foreach(TEST_CC_SRC ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ${LIBRARIES} OpenMeshRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

