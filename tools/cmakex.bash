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

See OPTICKS-


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

   git clone https://github.com/toomuchatonce/cmake-examples
}
