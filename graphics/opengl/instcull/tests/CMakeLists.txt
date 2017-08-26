cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name INSTCULLTest)
project(${name})

set(TEST_SOURCES
    txf.cc
    txfStream.cc
    instcull1.cc
    icdemo.cc
    instanceMinimal.cc
    instElemUBO.cc
    onetriangleMinimal.cc
    onetriangleElementMinimal.cc
    primRender.cc
    oneCubeMinimal.cc
    instanceVA.cc
    CompTest.cc
    GeomTest.cc
    BBTest.cc
    PrimTest.cc
    BoxTest.cc
    CamTest.cc
    VueTest.cc
    TraTest.cc
)
 
foreach(TEST_SOURCE ${TEST_SOURCES})

    get_filename_component(TGT ${TEST_SOURCE} NAME_WE)

    add_executable(${TGT} ${TEST_SOURCE})

    target_link_libraries(${TGT} ${LIBRARIES} INSTCULL)

    install(TARGETS ${TGT} DESTINATION lib)

endforeach()

