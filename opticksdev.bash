opticksdev-vi(){ vi $BASH_SOURCE ; }
opticksdev-usage(){ cat << \EOU

Opticks Dev Notes
====================

Aiming for opticks.bash to go in top level of a new Opticks repo
together with top level superbuild CMakeLists.txt
Intend to allow building independent of the env.

See Also
----------

cmake-
    background on cmake

cmakex-
    documenting the development of the opticks- cmake machinery 

cmakecheck-
    testing CMake config


How to add an external to Opticks
-----------------------------------

Opticks externals need to provide a name.bash 
that follows certain conventions regarding the bash functions
that it contains and some required functions:

name-
    required precursor, defines all the other functions

name--
    required rerunnable getter, builder and installer

name-url 
    optional distribution URL for info purposes

name-dist
    optional path to local distribution zip, tarball or clone/checkout directory for info purposes



Create the .bash in ~/opticks/exterals:

   cd ~/opticks/externals
   cp gleq.bash oimplicitmesher.bash   # gleq.bash is good template as very simple

Hookup the precursor function in ~/opticks/externals/externals.bash this
is sourced by the opticks- precursor adding a line like::

   oimplicitmesher-(){  . $(opticks-home)/externals/oimplicitmesher.bash   && oimplicitmesher-env $* ; }


Follow the standard opticks locations for source and build dirs and install prefixes as 
develop the name-get name-cmake and name-make functions, test rerunnability::

    delta:implicitmesher.build blyth$ oimplicitmesher--
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /usr/local/opticks/externals/implicitmesher/implicitmesher.build
    Scanning dependencies of target ImplicitMesher
    [ 53%] Built target ImplicitMesher
    [ 61%] Built target GenericFunctionFTest
    [ 69%] Built target GenericFunctionTest
    [ 76%] Built target ImplicitMesherFTest
    [ 84%] Built target ImplicitMesherTest
    [ 92%] Built target ImplicitPolygonizerTest
    [100%] Built target SimpleMeshTest
    Install the project...
    -- Install configuration: "Debug"
    -- Up-to-date: /usr/local/opticks/externals/lib/libImplicitMesher.dylib
    -- Up-to-date: /usr/local/opticks/externals/include/ImplicitMesher/ImplicitMesherF.h
    -- Up-to-date: /usr/local/opticks/externals/include/ImplicitMesher/ImplicitMesherBase.h
    -- Up-to-date: /usr/local/opticks/externals/lib/ImplicitPolygonizerTest
    -- Up-to-date: /usr/local/opticks/externals/lib/SimpleMeshTest
    -- Up-to-date: /usr/local/opticks/externals/lib/GenericFunctionTest
    -- Up-to-date: /usr/local/opticks/externals/lib/GenericFunctionFTest
    -- Up-to-date: /usr/local/opticks/externals/lib/ImplicitMesherTest
    -- Up-to-date: /usr/local/opticks/externals/lib/ImplicitMesherFTest
    delta:implicitmesher.build blyth$ 
     

Make the external usable using cmake find mechanism, adding ~/opticks/cmake/Modules/FindImplicitMesher.cmake::

    set(ImplicitMesher_PREFIX "${OPTICKS_PREFIX}/externals")

    find_library( ImplicitMesher_LIBRARIES 
                  NAMES ImplicitMesher
                  PATHS ${ImplicitMesher_PREFIX}/lib )

    set(ImplicitMesher_INCLUDE_DIRS "${ImplicitMesher_PREFIX}/include")
    set(ImplicitMesher_DEFINITIONS "")





Visibility inconsistency
--------------------------

::

    [ 75%] Building CXX object optickscore/CMakeFiles/OpticksCore.dir/Demo.cc.o
    [ 87%] Building CXX object optickscore/CMakeFiles/OpticksCore.dir/DemoCfg.cc.o
    [ 87%] Linking CXX shared library libOpticksCore.dylib
    ld: warning: direct access in boost::program_options::typed_value<std::__1::vector<int, std::__1::allocator<int> >, char>::value_type() const to global weak symbol typeinfo for std::__1::vector<int, std::__1::allocator<int> > means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in boost::typeindex::stl_type_index boost::typeindex::stl_type_index::type_id<std::__1::vector<int, std::__1::allocator<int> > >() to global weak symbol typeinfo for std::__1::vector<int, std::__1::allocator<int> > means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    [ 87%] Built target OpticksCore
    [ 87%] Linking CXX executable AnimatorTest




Fullbuild Testing
------------------

Only needed whilst making sweeping changes::

    simon:~ blyth$ opticks-distclean         # check what will be deleted
    simon:~ blyth$ opticks-distclean | sh    # delete 

    simon:~ blyth$ opticks-fullclean         # check what will be deleted
    simon:~ blyth$ opticks-fullclean | sh    # delete 

    simon:~ blyth$ opticks- ; opticks-full


Locating Boost, CUDA, OptiX
------------------------------

CMake itself provides cross platform machinery to find Boost and CUDA::

   /opt/local/share/cmake-3.4/Modules/FindBoost.cmake
   /opt/local/share/cmake-3.4/Modules/FindCUDA.cmake 

OptiX provides eg::

   /Developer/OptiX_380/SDK/CMake/FindOptiX.cmake

Tis self contained so copy into cmake/Modules
to avoid having to set CMAKE_MODULE_PATH to find it.  
This provides cache variable OptiX_INSTALL_DIR.


TODO
-----

* find out what depends on ssl and crypt : maybe in NPY_LIBRARIES 
* tidy up optix optixu FindOptiX from the SDK doesnt set OPTIX_LIBRARIES

* get the CTest tests to pass 

* incorporate cfg4- in superbuild with G4 checking

* check OptiX 4.0 beta for cmake changes 
* externalize or somehow exclude from standard building the Rap pkgs, as fairly stable
* look into isolating Assimp dependency usage

* spawn Opticks repository 
* adopt single level directories 
* split ggv- usage from ggeoview- building
* rename GGeoView to OpticksView/OpticksViz
* rename GGeo to OpticksGeo ?

* investigate CPack as way of distributing binaries


TODO: are the CUDA flags being used
------------------------------------

::

    simon:env blyth$ optix-cuda-nvcc-flags
    -ccbin /usr/bin/clang --use_fast_math


TODO: make envvar usage optional
----------------------------------

Enable all envvar settings to come in via the metadata .ini approach with 
envvars being used to optionally override those.

Bash launcher ggv.sh tied into the individual bash functions and 
sets up envvars::

   OPTICKS_GEOKEY
   OPTICKS_QUERY
   OPTICKS_CTRL
   OPTICKS_MESHFIX
   OPTICKS_MESHFIX_CFG

TODO:cleaner curand state
---------------------------

File level interaction between optixrap- and cudarap- 
in order to persist the state currently communicates via envvar ?

:: 

    simon:~ blyth$ l /usr/local/env/graphics/ggeoview/cache/rng
    total 344640
    -rw-r--r--  1 blyth  staff   44000000 Dec 29 20:33 cuRANDWrapper_1000000_0_0.bin
    -rw-r--r--  1 blyth  staff     450560 May 17  2015 cuRANDWrapper_10240_0_0.bin
    -rw-r--r--  1 blyth  staff  132000000 May 17  2015 cuRANDWrapper_3000000_0_0.bin


EOU
}

