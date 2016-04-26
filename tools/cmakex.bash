# === func-gen- : tools/cmakex fgp tools/cmakex.bash fgn cmakex fgh tools
cmakex-src(){      echo tools/cmakex.bash ; }
cmakex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cmakex-src)} ; }
cmakex-vi(){       vi $(cmakex-source) ; }
cmakex-env(){      elocal- ; }
cmakex-usage(){ cat << EOU

CMAKE Examples
================

staticlibs-add_subdir
-----------------------

* note linking with bare target name from another subdir 
  the library and needed dependent build is done automatically

* individual subdir build fails, but can build subdir libs
  from top level::

    simon:build blyth$ make help
    The following are some of the valid targets for this Makefile:
    ... all (the default if no target is provided)
    ... clean
    ... depend
    ... edit_cache
    ... rebuild_cache
    ... finally
    ... a
    ... b
    ... c
    ... main.o
    ... main.i
    ... main.s


OPTICKS MACHINERY DEV NOTES
=============================

See opticks-


CTest
------

* https://cmake.org/Wiki/CMake/Testing_With_CTest

Testing machinery operational with very little effort.

Configure testing
~~~~~~~~~~~~~~~~~~~

::

      1 cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
      2 set(name NPYTest)
      3 project(${name})
      4 
      5 file(GLOB TEST_CC_SRCS "*Test.cc")
      6 
      7 foreach(TEST_CC_SRC ${TEST_CC_SRCS})
      8     get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
      9     add_executable(${TGT} ${TEST_CC_SRC})
     10 
     11     # https://cmake.org/Wiki/CMakeEmulateMakeCheck
     12     add_test(${TGT} ${TGT})
     13     add_dependencies(check ${TGT})
     14 
     15     target_link_libraries(${TGT} 
     16                ${LIBRARIES} 
     17                NPY
     18     )
     19     install(TARGETS ${TGT} DESTINATION bin)
     20 endforeach()


Running tests
~~~~~~~~~~~~~~~~~

::

    Scanning dependencies of target check
    Test project /usr/local/opticks/build
     ...  
     
    78% tests passed, 8 tests failed out of 37

    Total Test time (real) =   6.30 sec

    The following tests FAILED:
          1 - _BoundariesNPYTest (SEGFAULT)
          2 - _LookupTest (OTHER_FAULT)
          3 - _RecordsNPYTest (SEGFAULT)
         11 - IndexTest (SEGFAULT)
         28 - PhotonsNPYTest (SEGFAULT)
         29 - readFlagsTest (Failed)
         31 - SequenceNPYTest (SEGFAULT)
         36 - TypesTest (SEGFAULT)
    Errors while running CTest
    make: *** [test] Error 8

Rebuild single test::

   opticks-bcd
   make readFlagsTest 

Retest single test::

    opticks-bcd

    simon:build blyth$ ctest -R readFlagsTest -V
    UpdateCTestConfiguration  from :/usr/local/opticks/build/DartConfiguration.tcl
    Parse Config file:/usr/local/opticks/build/DartConfiguration.tcl
    UpdateCTestConfiguration  from :/usr/local/opticks/build/DartConfiguration.tcl
    Parse Config file:/usr/local/opticks/build/DartConfiguration.tcl
    Test project /usr/local/opticks/build
    Constructing a list of tests
    Done constructing a list of tests
    Checking test dependency graph...
    Checking test dependency graph end
    test 29
        Start 29: readFlagsTest

    29: Test command: /usr/local/opticks/build/numerics/npy/tests/readFlagsTest
    29: Test timeout computed to be: 1500
    29: /usr/local/opticks/build/numerics/npy/tests/readFlagsTest missing input file /tmp/GFlagIndexLocal.ini
    1/1 Test #29: readFlagsTest ....................***Failed    0.01 sec

    0% tests passed, 1 tests failed out of 1

    Total Test time (real) =   0.01 sec

    The following tests FAILED:
         29 - readFlagsTest (Failed)
    Errors while running CTest
    simon:build blyth$ 



CPack
------

::

    [100%] Built target GGeoView
    Run CPack packaging tool...
    CPack: Create package using STGZ
    CPack: Install projects
    CPack: - Run preinstall target for: OPTICKS
    CPack: - Install project: OPTICKS
    CPack: Create package
    CPack: - package: /usr/local/opticks/build/OPTICKS-0.1.1-Darwin.sh generated.
    CPack: Create package using TGZ
    CPack: Install projects
    CPack: - Run preinstall target for: OPTICKS
    CPack: - Install project: OPTICKS
    CPack: Create package
    CPack: - package: /usr/local/opticks/build/OPTICKS-0.1.1-Darwin.tar.gz generated.


Creates a self extracting achive script::

    simon:build blyth$ l OPTICKS*
    -rw-r--r--  1 blyth  staff  5782311 Apr 25 15:32 OPTICKS-0.1.1-Darwin.tar.gz
    -rwxrwxrwx  1 blyth  staff  5785140 Apr 25 15:32 OPTICKS-0.1.1-Darwin.sh
    
    simon:build blyth$ ./OPTICKS-0.1.1-Darwin.sh --help
    Usage: ./OPTICKS-0.1.1-Darwin.sh [options]
    Options: [defaults in brackets after descriptions]
      --help            print this message
      --prefix=dir      directory in which to install
      --include-subdir  include the OPTICKS-0.1.1-Darwin subdirectory
      --exclude-subdir  exclude the OPTICKS-0.1.1-Darwin subdirectory
    simon:build blyth$ 

::

    simon:build blyth$ tar ztvf OPTICKS-0.1.1-Darwin.tar.gz
    drwxr-xr-x  0 blyth  staff       0 Apr 25 15:32 OPTICKS-0.1.1-Darwin/bin/
    -rwxr-xr-x  0 blyth  staff   21660 Apr 25 15:31 OPTICKS-0.1.1-Darwin/bin/cuRANDWrapperTest
    -rwxr-xr-x  0 blyth  staff   58460 Apr 25 15:32 OPTICKS-0.1.1-Darwin/bin/GGeoView
    -rwxr-xr-x  0 blyth  staff   27416 Apr 25 15:31 OPTICKS-0.1.1-Darwin/bin/LaunchSequenceTest
    drwxr-xr-x  0 blyth  staff       0 Apr 25 15:32 OPTICKS-0.1.1-Darwin/gl/
    drwxr-xr-x  0 blyth  staff       0 Apr 25 15:32 OPTICKS-0.1.1-Darwin/gl/altrec/
    -rw-r--r--  0 blyth  staff      98 Sep 14  2015 OPTICKS-0.1.1-Darwin/gl/altrec/frag.glsl
    -rw-r--r--  0 blyth  staff    1892 Sep 17  2015 OPTICKS-0.1.1-Darwin/gl/altrec/geom.glsl
    -rw-r--r--  0 blyth  staff     454 Sep 14  2015 OPTICKS-0.1.1-Darwin/gl/altrec/vert.glsl
    drwxr-xr-x  0 blyth  staff       0 Apr 25 15:32 OPTICKS-0.1.1-Darwin/gl/axis/
    -rw-r--r--  0 blyth  staff     111 Sep 14  2015 OPTICKS-0.1.1-Darwin/gl/axis/frag.glsl





RPATH/library confusion
-------------------------

RPATH covers all dirs including the collective, 
unclear which libs are getting used::

    simon:env blyth$ otool-;otool-rpath /usr/local/opticks/bin/GGeoView | grep path | uniq
         path /usr/local/cuda/lib (offset 12)
         path /usr/local/opticks/lib (offset 12)
         path /opt/local/lib (offset 12)
         path /usr/local/env/graphics/glew/1.12.0/lib (offset 12)
         path /usr/local/env/boost/bpo/bcfg/lib (offset 12)
         path /usr/local/env/boost/bregex/lib (offset 12)
         path /usr/local/env/numerics/npy/lib (offset 12)
         path /usr/local/env/opticks/lib (offset 12)
         path /usr/local/env/graphics/assimpwrap/lib (offset 12)
         path /usr/local/env/graphics/OpenMesh/4.1/lib (offset 12)
         path /usr/local/env/graphics/openmeshrap/lib (offset 12)
         path /usr/local/env/optix/ggeo/lib (offset 12)
         path /usr/local/env/graphics/gui/imgui.install/lib (offset 12)
         path /usr/local/env/graphics/oglrap/lib (offset 12)
         path /usr/local/env/cuda/CUDAWrap/lib (offset 12)
         path /usr/local/env/graphics/OptiXRap/lib (offset 12)
         path /usr/local/env/numerics/ThrustRap/lib (offset 12)
         path /usr/local/env/opticksop/lib (offset 12)
         path /usr/local/env/opticksgl/lib (offset 12)
         path /Developer/OptiX/lib64 (offset 12)


Initially:

* collective build installs everything to  -DCMAKE_INSTALL_PREFIX=$(local-base)/opticks 
* individual builds install many places eg  -DCMAKE_INSTALL_PREFIX=$(local-base)/env/boost/bpo/bcfg  etc...
* FindX.cmake returns the individual locations


Centralized approach ?
~~~~~~~~~~~~~~~~~~~~~~~~

* adopt centralized location for individual builds...

  * change all the FindX.cmake to give centralized position
  * nice and simple
  * BUT: means common namespace, so should improve classname prefixing  

* individual and collective builds that operate in the same pot 

* how to handle internal/external distinction ? which changes as pkg matured


Collective needs installs
--------------------------

::

    simon:env blyth$ OPTICKS-make
    [  1%] Built target Cfg
    [  2%] Built target Bregex
    [  4%] Built target regexsearchTest
    [  4%] Building CXX object numerics/npy/CMakeFiles/NPY.dir/NPYBase.cpp.o
    /Users/blyth/env/numerics/npy/NPYBase.cpp:13:10: fatal error: 'regexsearch.hh' file not found
    #include "regexsearch.hh"
             ^
    1 error generated.
    make[2]: *** [numerics/npy/CMakeFiles/NPY.dir/NPYBase.cpp.o] Error 1
    make[1]: *** [numerics/npy/CMakeFiles/NPY.dir/all] Error 2
    make: *** [all] Error 2
    simon:env blyth$ 


NPY was needing Bregex installed headers, so it fails on first run ?

* Workarounds look complicated, see cmake-
* Pragmatically adjust all internal _INCLUDE_DIRS to source directories rather than install directories. 
* That seemed to work for a while, but after a wipe OPTICKS-cmake fails as all the _LIBRARIES 
  come back NOTFOUND at config time


find_package assumes do not need to build/install it ? 
---------------------------------------------------------

Workaround this using SUPERBUILD variable, as explained in FindBregex.cmake::

    if(SUPERBUILD)
        if(NOT Bregex_LIBRARIES)
           set(Bregex_LIBRARIES Bregex)
        endif()
    endif(SUPERBUILD)

    # When no lib is found at configure time : ie when cmake is run
    # find_package normally yields NOTFOUND
    # but here if SUPERBUILD is defined Bregex_LIBRARIES
    # is set to the target name: Bregex. 
    # 
    # This allows the build to proceed if the target
    # is included amongst the add_subdirectory of the super build.
    #

Hmm getting cycle warnings
----------------------------

All due to interloper directory /usr/local/env/numerics/npy/lib::

    NOT WITH_NPYSERVER
    -- Configuring done
    CMake Warning at optix/ggeo/CMakeLists.txt:51 (add_library):
      Cannot generate a safe runtime search path for target GGeo because there is
      a cycle in the constraint graph:

        dir 0 is [/opt/local/lib]
        dir 1 is [/usr/local/env/numerics/npy/lib]
          dir 4 must precede it due to runtime library [libNPY.dylib]
        dir 2 is [/usr/local/opticks/build/ALL/opticks]
        dir 3 is [/usr/local/opticks/build/ALL/boost/bpo/bcfg]
        dir 4 is [/usr/local/opticks/build/ALL/numerics/npy]
          dir 1 must precede it due to runtime library [libNPY.dylib]
        dir 5 is [/usr/local/opticks/build/ALL/boost/bregex]

      Some of these libraries may not be found correctly.

::

    simon:env blyth$ otool-;otool-rpath /usr/local/opticks/bin/GGeoView  | grep path | uniq
         path /usr/local/cuda/lib (offset 12)
         path /usr/local/opticks/lib (offset 12)
         path /opt/local/lib (offset 12)
         path /usr/local/env/graphics/glew/1.12.0/lib (offset 12)
         path /usr/local/env/numerics/npy/lib (offset 12)
         path /usr/local/env/graphics/OpenMesh/4.1/lib (offset 12)
         path /usr/local/env/graphics/gui/imgui.install/lib (offset 12)
         path /Developer/OptiX/lib64 (offset 12)


This issue went away, pilot error ?


Install seems to build again ?
-----------------------------------

::

  614  rm -rf /usr/local/opticks/*
  615  OPTICKS-
  616  OPTICKS-cmake
  617  OPTICKS-make
  625  OPTICKS-install
  626  otool-
  627  otool-rpath /usr/local/opticks/bin/GGeoView 
  628  /usr/local/opticks/bin/GGeoView /tmp/g4_00.dae


Observations
--------------

subbuild bash funcs needed ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wandering around the OPTICKS-bcd it is apparent that 
do not really need bash functions for individual pkg builds
once the superbuild is working. Could make the 
individual builds cd into the superbuild build dir and make 
from there.

::

    simon:bregex blyth$ make help
    The following are some of the valid targets for this Makefile:
    ... all (the default if no target is provided)
    ... clean
    ... depend
    ... edit_cache
    ... Bregex


tendency to overrun cmake
~~~~~~~~~~~~~~~~~~~~~~~~~~

Can operate with much fewer cmake invokations, 
normal devshould not need to configure (ie run cmake) very much.  

Occasional configuration, on adding new sources::

   OPTICKS-cmake

Debug cycle becomes:

* edit sources
* invoke make/install from appropriate superbuild build directory 
  to remake single library
* OR for changes to many libs, invoke make/install from top level
  of superbuild
* run test binaries


too many binaries
~~~~~~~~~~~~~~~~~~~

All tests are bundled into /usr/local/opticks/bin/






EOU
}
cmakex-dir(){ echo $(local-base)/env/tools/cmake-examples ; }
cmakex-cd(){  cd $(cmakex-dir); }
cmakex-get(){
   local dir=$(dirname $(cmakex-dir)) &&  mkdir -p $dir && cd $dir

   #git clone https://github.com/toomuchatonce/cmake-examples
   git clone -v https://github.com/simoncblyth/cmake-examples
}


cmakex-wipe(){
   local iwd=$PWD
   local dir=$(dirname $(cmakex-dir)) &&  mkdir -p $dir && cd $dir

   [ -d cmake-examples ] && rm -rf cmake-examples
   
   cd $iwd
}

