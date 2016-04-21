# === func-gen- : tools/cmake fgp tools/cmake.bash fgn cmake fgh tools
cmake-src(){      echo tools/cmake.bash ; }
cmake-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cmake-src)} ; }
cmake-vi(){       vi $(cmake-source) ; }
cmake-env(){      elocal- ; }
cmake-usage(){ cat << EOU

CMAKE
======

* http://www.cmake.org/


Issues
-------

* note that refactoring that removes a header from a package 
  is not noticed and cleared from a install location resulting
  in old headers being left in include folders 

  * workaround is to occasionally delete the include folder
    and rebuild and reinstall


Superpackage using add_subdirectory with find_package in each
--------------------------------------------------------------

* :google:`cmake subdirectory with find_package`

  * https://cmake.org/pipermail/cmake/2013-April/054415.html 

  * http://mirkokiefer.com/blog/2013/03/cmake-by-example/

  * https://coderwall.com/p/y3zzbq/use-cmake-enabled-libraries-in-your-cmake-project

  * https://github.com/Athius/FrameworkCMakeToolkit


Using ExternalProject looks like overkill, maybe 
some devious relative install paths can get single project and 
superproject builds to work together ?


Policy control
---------------

::

    # from ggv-
    #cmake_policy(SET CMP0054 OLD)
    # unfortunately this doesnt suppress the warnings, despite being advertised to do so
    # http://www.cmake.org/Wiki/CMake/Policies



Best Docs encountered
----------------------

* http://www.cmake.org/cmake/help/git-master/command/find_path.html
* http://www.cmake.org/cmake/help/git-master/command/find_package.html

Other
------

rule for generated header files in sub directories

* http://www.cmake.org/pipermail/cmake/2012-November/052775.html

Tips 
-----

* http://web.cs.swarthmore.edu/~adanner/tips/cmake.php


Other Docs
------------

* http://www.cmake.org/cmake/help/cmake2.4docs.html
* http://www.cmake.org/Wiki/CMakeMacroListOperations


Random Projects with good cmake usage docs
--------------------------------------------

* https://software.sandia.gov/trac/dakota/wiki/CMakeFAQ

Finding Libs
------------

* http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries
* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/installconfig/1467.html?inline=-1


Macros
-------

* /usr/local/env/chroma_env/src/root-v5.34.14/cmake/modules/RootNewMacros.cmake



Versions
---------

::

    g4pb:~ blyth$ cmake -version
    cmake version 2.8.7

    [blyth@belle7 ~]$ cmake -version
    cmake version 2.6-patch 4

    delta:~ blyth$ cmake -version
    cmake version 2.8.12

    -bash-4.1$ cmake -version
    cmake version 2.6-patch 4

    delta:~ blyth$ which cmake
    /opt/local/bin/cmake


macports cmake
---------------

* https://trac.macports.org/browser/trunk/dports/devel/cmake/Portfile


::

    delta:~ blyth$ port info cmake
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    cmake @2.8.12.2 (devel)
    Variants:             gui, universal

    Description:          An extensible, open-source system that manages the build process in an operating system and compiler independent manner. Unlike many cross-platform systems, CMake is designed to be used in
                          conjunction with the native build environment.
    Homepage:             http://www.cmake.org/

    Library Dependencies: libidn, openssl
    Platforms:            darwin, freebsd
    License:              BSD
    Maintainers:          css@macports.org
    delta:~ blyth$ 




Debugging
----------
::

    cmake --trace .
    cmake -DCMAKE_BUILD_TYPE:STRING=Debug 

::

    make VERBOSE=1



avoiding cmake full builds on rerunning
------------------------------------------

* http://stackoverflow.com/questions/8479929/cmake-add-subdirectory-and-recompiling



cmake dumping
-------------

Untried from osgPlugins::

    ##########to get all the variables of Cmake
    #GET_CMAKE_PROPERTY(MYVARS VARIABLES)
    #FOREACH(myvar ${MYVARS})
    #    FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/AllVariables.txt
    #        "${myvar} -->${${myvar}}<-\n"
    #    )
    #ENDFOREACH(myvar)


cmake architectures
------------------------

* http://stackoverflow.com/questions/5334095/cmake-multiarchitecture-compilation



Makefile Debug
---------------
::

    set(CMAKE_VERBOSE_MAKEFILE ON)

From Scratch Build
--------------------

Build approach that does everything in one, avoiding all caching is useful for CMakeLists.txt iteration::

    ( rm -rf build ; mkdir build ; cd build ; cmake .. ; make )

cmake usage examples
-----------------------

* ~/e/graphics/collada/colladadom/testColladaDOM/CMakeLists.txt


Help
-----

::

    cmake --help-command LIST




EXTERNAL LIBS
--------------

* http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries
* http://www.cmake.org/cmake/help/v2.8.8/cmake.html#command:find_package

::

    [blyth@belle7 gdml2wrl]$ cmake --help-module-list | grep Find
    CMakeFindFrameworks
    FindASPELL
    FindAVIFile
    FindBLAS
    FindBZip2
    FindBoost
    FindCABLE
    FindCURL
    FindCVS
    FindCoin3D


Geant4.cmake
~~~~~~~~~~~~~

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/ch03s02.html


Finding Root
~~~~~~~~~~~~~~

Nope::

    (chroma_env)delta:LXe blyth$ cmake --help-module-list | grep ROOT
    (chroma_env)delta:LXe blyth$ 

::

    (chroma_env)delta:LXe blyth$ mdfind FindROOT.cmake
    /usr/local/env/geant4/geant4.10.00.p01/cmake/Modules/FindROOT.cmake
    /usr/local/env/geant4/geant4.10.00.p01/environments/g4py/cmake/Modules/FindROOT.cmake
    /usr/local/env/chroma_env/src/geant4.9.5.p01/cmake/Modules/FindROOT.cmake
    /usr/local/env/chroma_env/src/root-v5.34.11/etc/cmake/FindROOT.cmake
    /usr/local/env/chroma_env/src/root-v5.34.14/etc/cmake/FindROOT.cmake
    /usr/local/env/chroma_env/src/root-v5.34.14.patch01/etc/cmake/FindROOT.cmake



CMAKE OSX RPATH INSTALL_NAME_TOOL ISSUE
-----------------------------------------

* :google:`cmake linking rpath install_name_tool`

::

    (chroma_env)delta:zmqroot-build blyth$ cmake -version
    cmake version 2.8.12

    Install the project...
    /opt/local/bin/cmake -P cmake_install.cmake
    -- Install configuration: ""
    -- Up-to-date: /usr/local/env/zmqroot/include/ZMQRoot.hh
    -- Installing: /usr/local/env/zmqroot/lib/libZMQRoot.dylib
    /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/install_name_tool: 
            object: /usr/local/env/zmqroot/lib/libZMQRoot.dylib malformed object (load command 32 cmdsize is zero)
    (chroma_env)delta:zmqroot-build blyth$ 


Linking repeats rpath::

     /usr/bin/c++   
       -dynamiclib -Wl,-headerpad_max_install_names 
       -o libZMQRoot.dylib 
       -install_name /tmp/env/zmqroot-build/libZMQRoot.dylib 
            CMakeFiles/ZMQRoot.dir/src/MyTMessage.cc.o 
            CMakeFiles/ZMQRoot.dir/src/ZMQRoot.cc.o 
            CMakeFiles/ZMQRoot.dir/MyTMessageDict.cxx.o 
       -L/usr/local/env/chroma_env/src/root-v5.34.14/lib 
       -L/usr/local/env/chroma_env/lib 
       -L/usr/local/env/chroma_env/src/root-v5.34.14/lib
              -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lpthread
        -Wl,-rpath,/usr/local/env/chroma_env/src/root-v5.34.14/lib
        -stdlib=libc++ -lm -ldl -lzmq
        -Wl,-rpath,/usr/local/env/chroma_env/src/root-v5.34.14/lib
        -Wl,-rpath,/usr/local/env/chroma_env/lib 


Below ticket suggests repetition of rpath passed to linker is the cause of the issue

* https://github.com/SimTk/openmm/issues/295

Full details/workaround in below ticket (removing link_directories) 

* http://public.kitware.com/Bug/view.php?id=14707

cmake rpath handling 
----------------------

* http://www.kitware.com/blog/home/post/510
* http://www.cmake.org/Wiki/CMake_RPATH_handling
* http://stackoverflow.com/questions/9263256/can-you-please-help-me-understand-how-mach-o-libraries-work-in-mac-os-x
* http://www.dribin.org/dave/blog/archives/2009/11/15/rpath/
* https://mikeash.com/pyblog/friday-qa-2009-11-06-linking-and-install-names.html
* http://matthew-brett.github.io/pydagogue/mac_runtime_link.html


root example
------------

/usr/local/env/chroma_env/src/root-v5.34.14/cmake/modules/RootBuildOptions.cmake


EOU
}
cmake-dir(){ echo $(local-base)/env/tools/tools-cmake ; }
cmake-cd(){  cd $(cmake-dir); }
cmake-mate(){ mate $(cmake-dir) ; }
cmake-get(){
   local dir=$(dirname $(cmake-dir)) &&  mkdir -p $dir && cd $dir

}

cmake-find(){
   cd $ENV_HOME
   find . -name CMakeLists.txt -exec grep -H ${1:-ZMQ} {} \;
}

