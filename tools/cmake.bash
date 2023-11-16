# === func-gen- : tools/cmake fgp tools/cmake.bash fgn cmake fgh tools
cmake-src(){      echo tools/cmake.bash ; }
cmake-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cmake-src)} ; }
cmake-vi(){       vi $(cmake-source) ; }
cmake-env(){      elocal- ; }
cmake-usage(){ cat << EOU

CMAKE
======

* http://www.cmake.org/
* https://cmake.org/download/

* https://gitlab.kitware.com/cmake/cmake/commit/874fdd914a646d25096c34b97caafe43e2a77748



Creating an umbrella cmake config for Opticks ? 
-------------------------------------------------

Problem with this is that its very much a virtual package 
as there is no top level Opticks CMakeLists.txt its a 
bunch of packages strapped together with bash. 

* maybe with BCM ? 

* https://iamsorush.com/posts/cpp-cmake-config/
* https://cmake.org/cmake/help/latest/module/CMakePackageConfigHelpers.html
* https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html

Custom4 takes a minimal approach at doing this : probably a 
good approach, BCM is too involved.  Could generate frm G4CX the 
the top level package. 

See /usr/local/opticks_externals/custom4/0.1.9/lib/Custom4-0.1.9


LOOKS LIKE A GOOD CMAKE COURSE
--------------------------------

* https://enccs.github.io/cmake-workshop/
* https://github.com/ENCCS/cmake-workshop

* https://github.com/ENCCS/cmake-workshop/blob/main/content/code/day-1/06_bash-ctest/solution/CMakeLists.txt


EuroCC National Competence Center Sweden 
-------------------------------------------

* https://github.com/ENCCS
* https://github.com/orgs/ENCCS/repositories?type=all

* https://enccs.se/lessons/


Manual Target Deployment is just too ugly : use BCMExport
------------------------------------------------------------

::

    #[=[
    https://cmake.org/cmake/help/git-master/guide/tutorial/index.html#adding-export-configuration-step-11
    https://cmake.org/cmake/help/git-stage/guide/importing-exporting/index.html

    set(CONFIG_DEST ${CMAKE_INSTALL_LIBDIR}/cmake/CSG) 
    set(CSG_INCLUDE_DIRS "include")
    install(TARGETS ${name}  DESTINATION ${CMAKE_INSTALL_LIBDIR} EXPORT CSGTargets )
    install(EXPORT CSGTargets FILE CSGTargets.cmake DESTINATION ${CONFIG_DEST}) 
    include(CMakePackageConfigHelpers)
    configure_package_config_file ( 
           "CSGConfig.cmake.in" "${CMAKE_BINARY_DIR}/CSGConfig.cmake"
           INSTALL_DESTINATION "${CONFIG_DEST}"
           PATH_VARS CSG_INCLUDE_DIRS 
    )
    install(FILES ${CMAKE_BINARY_DIR}/CSGConfig.cmake DESTINATION ${CONFIG_DEST})
    #]=]





See Also
--------

cmakex-  
    cmake examples and documenting the development of OPTICKS- cmake machinery 

cmak-
    generating CMakeLists.txt to debug finding packages and flag setup etc.. 


Epsilon : macOS 10.13.4 : cmake @3.11.0
------------------------------------------

::

    sudo port install -v cmake +docs +python27



CMake find_package implementation
------------------------------------

* https://github.com/Kitware/CMake/blob/master/Source/cmFindPackageCommand.cxx


configure_package_config_file
---------------------------------

* https://cmake.org/cmake/help/latest/module/CMakePackageConfigHelpers.html

* https://gitlab.kitware.com/cmake/cmake/-/issues/19560

* https://gitlab.kitware.com/cmake/cmake/-/issues/17282


cmake import export
-----------------------

* https://cmake.org/cmake/help/git-stage/guide/importing-exporting/index.html



target_sources
----------------

* https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/



target_link_libraries on imported target : suspect CMake support varies between recent versions
--------------------------------------------------------------------------------------------------

* ~/opticks/notes/issues/cmake_target_link_libraries_for_imported_target.rst
* https://stackoverflow.com/questions/36648375/what-are-the-differences-between-imported-target-and-interface-libraries



add_custom_target
--------------------

* https://stackoverflow.com/questions/35300833/cmake-get-output-of-some-command-every-build-and-reconfigure-files-that-depends
* https://cmake.org/cmake/help/v3.0/command/add_custom_target.html

Adds a target with the given name that executes the given commands. The target
has no output file and is ALWAYS CONSIDERED OUT OF DATE even if the commands
try to create a file with the name of the target. Use ADD_CUSTOM_COMMAND to
generate a file with dependencies. By default nothing depends on the custom
target. Use ADD_DEPENDENCIES to add dependencies to or from other targets. If
the ALL option is specified it indicates that this target should be added to
the default build target so that it will be run every time (the command cannot
be called ALL). The command and arguments are optional and if not specified an
empty target will be created. If WORKING_DIRECTORY is set, then the command
will be run in that directory. If it is a relative path it will be interpreted
relative to the build tree directory corresponding to the current source
directory. If COMMENT is set, the value will be displayed as a message before
the commands are executed at build time. Dependencies listed with the DEPENDS
argument may reference files and outputs of custom commands created with
add_custom_command() in the same directory (CMakeLists.txt file).




cmake --build . --help
-------------------------

::

    Usage: cmake --build <dir> [options] [-- [native-options]]
    Options:
      <dir>          = Project binary directory to be built.
      --target <tgt> = Build <tgt> instead of default targets.
      --config <cfg> = For multi-configuration tools, choose <cfg>.
      --clean-first  = Build target 'clean' first, then build.
                       (To clean only, use --target 'clean'.)
      --use-stderr   = Ignored.  Behavior is default in CMake >= 3.0.
      --             = Pass remaining options to the native tool.



Best Docs encountered
----------------------

* http://www.cmake.org/cmake/help/git-master/command/find_path.html
* http://www.cmake.org/cmake/help/git-master/command/find_package.html
* https://cmake.org/cmake/help/git-master/manual/cmake-variables.7.html?highlight=cmake_install_prefix
* https://cmake.org/cmake/help/v3.0/command/add_dependencies.html
* https://cmake.org/cmake/help/v3.0/command/target_include_directories.html

* https://cmake.org/Wiki/CMake/Tutorials/Exporting_and_Importing_Targets
* http://www.kitware.com/media/html/BuildingExternalProjectsWithCMake2.8.html

* https://cmake.org/Wiki/CMake:How_To_Find_Libraries

Converting autotools to CMake ?
---------------------------------

Advice for how to convert, not easy.

* http://stackoverflow.com/questions/7132862/tutorial-for-converting-autotools-to-cmake
* http://www.vtk.org/Wiki/CMake#Converters_from_other_buildsystems_to_CMake
* https://sourceforge.net/p/vcproj2cmake/code/

Powershell parse vcproj XML and spit out simple CMakeLists.txt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://nberserk.blogspot.tw/2010/11/converting-vc-projectsvcproj-to.html

supports only VS2005 
* https://github.com/nberserk/common/blob/master/vcproj2cmake.ps1


CMake INTERFACE
------------------

* https://cmake.org/cmake/help/v3.0/prop_tgt/INTERFACE_INCLUDE_DIRECTORIES.html
* https://github.com/yeswalrus/cmake-modules

* http://stackoverflow.com/questions/26037954/cmake-target-link-libraries-interface-dependencies

If you are creating a shared library and your source cpp files #include the
headers of another library (Say, QtNetwork for example), but your header files
don't include QtNetwork headers, then QtNetwork is a PRIVATE dependency.

If your source files and your headers include the headers of another library,
then it is a PUBLIC dependency.

If your header files but not your source files include the headers of another
library, then it is an INTERFACE dependency.


Controlling Transitivity, ie the implicit passing along of dependencies 
-----------------------------------------------------------------------------

* https://cmake.org/cmake/help/v3.0/manual/cmake-buildsystem.7.html#transitive-usage-requirements


CMake 3.1 target_sources
---------------------------

* https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/


Unit Tests 
------------

* https://cmake.org/Wiki/CMakeEmulateMakeCheck
* http://stackoverflow.com/questions/14446495/cmake-project-structure-with-unit-tests


Experience : added test not running ? so touch tests/CMakeLists.txt
----------------------------------------------------------------------

Use of globbing in tests/CMakeLists.txt is convenient but sometimes
results in new tests not being noticed by the build.  Force it 
to be noticed the first time by touching the tests/CMakeLists.txt. 

This is why use of globbing is not a good idea for the primary sources,
but tis appropriate for tests as generally have lots of small sources
so it makes sense.



Experience
------------

* https://rix0r.nl/blog/2015/08/13/cmake-guide/

* https://github.com/toomuchatonce/cmake-examples/blob/master/superbuild-configtargets-direct/CMakeLists.txt


special flags for single files
--------------------------------

* http://stackoverflow.com/questions/13638408/cmake-override-compile-flags-for-single-files
* https://cmake.org/cmake/help/v3.3/manual/cmake-properties.7.html




envvars
--------

* https://cmake.org/Wiki/CMake_FAQ

One should avoid using environment variables for controlling the flow of CMake
code (such as in IF commands). The build system generated by CMake may re-run
CMake automatically when CMakeLists.txt files change. The environment in which
this is executed is controlled by the build system and may not match that in
which CMake was originally run. If you want to control build settings on the
CMake command line, you need to use cache variables set with the -D option. The
settings will be saved in CMakeCache.txt so that they don't have to be repeated
every time CMake is run on the same build tree.


cmake/make rebuilding for everytime
-------------------------------------

For example running "make package" 

* https://cmake.org/pipermail/cmake/2012-October/052525.html
* http://www.vtk.org/Wiki/CMake_Useful_Variables#Various_Options

::

   set (CMAKE_SKIP_RULE_DEPENDENCY TRUE)

CMAKE_SKIP_RULE_DEPENDENCY 
    set this to true if you don't want to rebuild the object files if the rules
    have changed, but not the actual source files or headers (e.g. if you changed
    the some compiler switches)



Introspection
---------------

::

    simon:env blyth$ OPTICKS-make -p
    ...gory details of the build...

    simon:env blyth$ OPTICKS-make help
    The following are some of the valid targets for this Makefile:
    ... all (the default if no target is provided)
    ... clean
    ... depend
    ... edit_cache
    ... rebuild_cache
    ... list_install_components
    ... install
    ... install/strip
    ... install/local
    ... Cfg
    ... Bregex
    ... os_path_expandvarsTest
    ... regexsearchTest
    ... regex_matched_elementTest
    ... regex_extract_quotedTest


add_subdirectory
----------------

* https://cmake.org/cmake/help/v2.8.8/cmake.html#command:add_subdirectory

If the EXCLUDE_FROM_ALL argument is provided then targets in the subdirectory
will not be included in the ALL target of the parent directory by default, and
will be excluded from IDE project files. Users must explicitly build targets in
the subdirectory. This is meant for use when the subdirectory contains a
separate part of the project that is useful but not necessary, such as a set of
examples. Typically the subdirectory should contain its own project() command
invocation so that a full build system will be generated in the subdirectory
(such as a VS IDE solution file). Note that inter-target dependencies supercede
this exclusion. If a target built by the parent project depends on a target in
the subdirectory, the dependee target will be included in the parent project
build system to satisfy the dependency.

target_link_libraries
----------------------

* :google:`cmake add_subdirectory find_package needs install`

* http://stackoverflow.com/questions/31755870/how-to-use-libraries-within-my-cmake-project-that-need-to-be-installed-first

* http://stackoverflow.com/questions/31537602/how-to-use-cmake-to-find-and-link-to-a-library-using-install-export-and-find-pac/31537603#31537603


* https://cmake.org/cmake/help/v3.0/command/target_link_libraries.html

The PUBLIC, PRIVATE and INTERFACE keywords can be used to specify both the link
dependencies and the link interface in one command. Libraries and targets
following PUBLIC are linked to, and are made part of the link interface.
Libraries and targets following PRIVATE are linked to, but are not made part of
the link interface. Libraries following INTERFACE are appended to the link
interface and are not used for linking <target>.




CMAKE_INSTALL_PREFIX
---------------------

* https://cmake.org/cmake/help/git-master/variable/CMAKE_INSTALL_PREFIX.html

...this directory is prepended onto all install directories. 
This variable defaults to /usr/local on UNIX and c:/Program Files on Windows.

The installation prefix is also added to CMAKE_SYSTEM_PREFIX_PATH so that
find_package(), find_program(), find_library(), find_path(), and find_file()
will search the prefix for other software.


LIBRARY_OUTPUT_DIRECTORY
--------------------------

* https://cmake.org/cmake/help/git-master/prop_tgt/LIBRARY_OUTPUT_DIRECTORY.html#prop_tgt:LIBRARY_OUTPUT_DIRECTORY



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


Where find_package looks
--------------------------

::

      CMake Warning at CMakeLists.txt:18 (find_package):
      By not providing "Findsniper.cmake" in CMAKE_MODULE_PATH this project has
      asked CMake to find a package configuration file provided by "sniper", but
      CMake did not find one.

      Could not find a package configuration file provided by "sniper" with any
      of the following names:

        sniperConfig.cmake
        sniper-config.cmake

      Add the installation prefix of "sniper" to CMAKE_PREFIX_PATH or set
      "sniper_DIR" to a directory containing one of the above files.  If "sniper"
      provides a separate development package or SDK, be sure it has been
      installed.





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



CMake RPATH Handling
------------------------

Starting from scratch with a simple CMakeList.txt on OSX get::

    delta:ImplicitMesher.build blyth$ implicitmesher-cmake
    -- Configuring done
    CMake Warning (dev):
      Policy CMP0042 is not set: MACOSX_RPATH is enabled by default.  Run "cmake
      --help-policy CMP0042" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      MACOSX_RPATH is not specified for the following targets:

       ImplicitMesher

    This warning is for project developers.  Use -Wno-dev to suppress it.





* https://cmake.org/Wiki/CMake_RPATH_handling


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


cmake-vers(){ echo v3.5 ; }
cmake-name(){ echo cmake-3.5.2-$(uname -s)-$(uname -m) ; }
cmake-dir(){ 
   case $NODE_TAG in 
       MGB) echo /c/ProgramData/chocolatey/lib/cmake.portable/tools/cmake-3.5.2-win32-x86 ;;
         *) echo $(local-base)/env/tools/cmake/$(cmake-name) ;;
   esac
}


cmake-hdir(){
   case $NODE_TAG in 
      D) echo /opt/local/share/cmake-3.4/Help ;;
   esac
}
cmake-hcd(){ cd $(cmake-hdir) ; }


cmake-find-package(){ 
   local pkg=${1:-Boost}
   echo $(cmake-dir)/share/cmake-3.5/Modules/Find${pkg}.cmake  
}


cmake-cd(){  cd $(cmake-dir); }
cmake-get(){

   [ "$NODE_TAG" == "MGB" ] && echo use chocolatey install on windows && return 


   local dir=$(dirname $(cmake-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(cmake-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz
}

cmake-url(){    echo https://cmake.org/files/$(cmake-vers)/$(cmake-name).tar.gz ; }
cmake-bindir(){ 
   case $(uname -s) in
     Darwin) echo $(cmake-dir)/CMake.app/Contents/bin ;;
      Linux) echo $(cmake-dir)/bin  ;;
          *) echo hmmm ;;
   esac 
}
cmake-alias(){  alias cmake352=$(cmake-bindir)/cmake ; }

cmake-export(){
   local bindir=$(cmake-bindir)
   if [ -d "$bindir" ]; then
       if [ "${PATH/$bindir}" == "$PATH" ]; then
           export PATH=$bindir:$PATH
       fi 
   fi
}

cmake-find(){
   cd $ENV_HOME
   find . -name CMakeLists.txt -exec grep -H ${1:-ZMQ} {} \;
}

