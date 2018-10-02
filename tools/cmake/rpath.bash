# === func-gen- : tools/cmake/rpath fgp tools/cmake/rpath.bash fgn rpath fgh tools/cmake src base/func.bash
rpath-source(){   echo ${BASH_SOURCE} ; }
rpath-edir(){ echo $(dirname $(rpath-source)) ; }
rpath-ecd(){  cd $(rpath-edir); }
rpath-dir(){  echo $LOCAL_BASE/env/tools/cmake/rpath ; }
rpath-cd(){   cd $(rpath-dir); }
rpath-vi(){   vi $(rpath-source) ; }
rpath-env(){  elocal- ; }
rpath-usage(){ cat << EOU



intro_to_cuda/useRecon::

  clang -I$(itc-prefix)/include -std=c++11 -L$(itc-prefix)/lib -lRecon -lc++ -Wl,-rpath $(itc-prefix)/lib useRecon.cc && ./a.out && rm a.out





* https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling

* https://stackoverflow.com/questions/30398238/cmake-rpath-not-working-could-not-find-shared-object-file

::

    SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
    SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_LIBDIR})
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


    set_target_properties(${name}Test PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_LIBDIR})
    set_target_properties(${name}Test PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(${name}Test PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    add_executable(${name}Test ${name}Test.cc)
    target_link_libraries(${name}Test PRIVATE ${name} ${CUDA_LIBRARIES} )
    install(TARGETS ${name}Test RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})


::

    cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
    set(name Recon)

    project(${name} VERSION 0.1.0 LANGUAGES CXX CUDA )

    #[=[
    Try to use modern CMake native CUDA support, following 

       https://devblogs.nvidia.com/building-cuda-applications-cmake/
    #]=]


    include(GNUInstallDirs)

    set(SOURCES 
        Recon.cu
    )

    set(HEADERS
        Recon.hh
    )

    add_library(${name} SHARED ${SOURCES})

    #target_compile_features(${name} PUBLIC cxx_std_11)
    #set_target_properties( ${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 


    add_executable(${name}Test ${name}Test.cc)
    target_link_libraries(${name}Test PRIVATE ${name})

    if(APPLE)
      # We need to add the path to the driver (libcuda.dylib) as an rpath, 
      # so that the static cuda runtime can find it at runtime.
      set_property(TARGET ${name}Test 
                   PROPERTY
                   BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()


    install(TARGETS ${name} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(TARGETS ${name}Test DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})




-- Up-to-date: /tmp/blyth/intro_to_cuda/lib/TimTest
error: /opt/local/bin/install_name_tool: no LC_RPATH load command with path: /tmp/blyth/intro_to_cuda/build/recon found in: /tmp/blyth/intro_to_cuda/lib/TimTest (for architecture x86_64), required for specified option "-delete_rpath /tmp/blyth/intro_to_cuda/build/recon"
error: /opt/local/bin/install_name_tool: for: /tmp/blyth/intro_to_cuda/lib/TimTest (for architecture x86_64) option "-add_rpath /tmp/blyth/intro_to_cuda/lib" would duplicate path, file already has LC_RPATH for: /tmp/blyth/intro_to_cuda/lib




EOU
}
rpath-get(){
   local dir=$(dirname $(rpath-dir)) &&  mkdir -p $dir && cd $dir

}
