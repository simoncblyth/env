cmak-src(){      echo tools/cmak.bash ; }
cmak-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cmak-src)} ; }
cmak-vi(){       vi $(cmak-source) ; }
cmak-env(){      elocal- ; }
cmak-usage(){ cat << \EOU


Release History
-------------------

* https://blog.kitware.com/cmake-3-11-2-available-for-download/

  CMake 3.11.2 available for download
  Robert Maynard on May 17, 2018

* https://blog.kitware.com/cmake-3-11-1-available-for-download/

  CMake 3.11.1 available for download
  Robert Maynard on April 17, 2018

* https://blog.kitware.com/cmake-3-11-0-available-for-download/

  CMake 3.11.0 available for download
  Robert Maynard on March 28, 2018

* https://blog.kitware.com/cmake-3-10-3-available-for-download/

  CMake 3.10.3 available for download
  Robert Maynard on March 16, 2018

* https://blog.kitware.com/cmake-3-10-2-available-for-download/

  CMake 3.10.2 available for download
  Robert Maynard on January 18, 2018


* https://cmake.org/files/



Reference
----------

* https://gitlab.kitware.com/cmake/community/wikis/home

CMake cxx features
-------------------

* https://cmake.org/cmake/help/v3.1/command/target_compile_features.html#command:target_compile_features
* https://cmake.org/cmake/help/v3.1/prop_gbl/CMAKE_CXX_KNOWN_FEATURES.html
* https://cmake.org/cmake/help/v3.1/manual/cmake-compile-features.7.html#manual:cmake-compile-features(7)


CMake RPATH handling 
----------------------

* https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling


CMake RPATH handling : kludge
-------------------------------

Flaky issue on Darwin, kludge fix is to sleep between "all" and "install"::

    -- Up-to-date: /usr/local/opticks-cmake-overhaul/lib/ygltf_reader
    error: /opt/local/bin/install_name_tool: no LC_RPATH load command with path: /usr/local/opticks-cmake-overhaul/externals/yoctogl/yocto-gl.build found in: /usr/local/opticks-cmake-overhaul/lib/ygltf_reader (for architecture x86_64), required for specified option "-delete_rpath /usr/local/opticks-cmake-overhaul/externals/yoctogl/yocto-gl.build"
    epsilon:yocto-gl blyth$ 


Mach-O rpath editing should check before performing its actions

* https://gitlab.kitware.com/cmake/cmake/issues/16155

Workaround is to sleep 2s between "all" and "install"::

    oyoctogl-- () 
    { 
        oyoctogl-get;
        oyoctogl-cmake;
        oyoctogl-make all;
        if [ "$(uname)" == "Darwin" ]; then
            echo sleeping for 2s : see env/tools/cmak.bash and https://gitlab.kitware.com/cmake/cmake/issues/16155;
            sleep 2;
        fi;
        oyoctogl-make install
    }



Examples
---------

* https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/Examples


CMake Properties
------------------

* https://cmake.org/cmake/help/v3.2/manual/cmake-properties.7.html?highlight=target%20properties#properties-on-targets


find_dependency from 3.10 the docs say it forwards additional args to find_package
------------------------------------------------------------------------------------

* seems to do so in 3.5 however

* https://cmake.org/cmake/help/v3.11/module/CMakeFindDependencyMacro.html

  Any additional arguments specified are forwarded to find_package(). 



FindBoost.cmake
-----------------

::

    epsilon:UseBoost blyth$ port contents cmake | grep Boost 
      /opt/local/share/cmake-3.11/Modules/FindBoost.cmake
      /opt/local/share/doc/cmake/html/module/FindBoost.html
    epsilon:UseBoost blyth$ 




FindPackage.cmake examples
----------------------------

::

    epsilon:opticks blyth$ port contents cmake | grep Modules/Find 
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
      /opt/local/share/cmake-3.11/Modules/FindALSA.cmake
      /opt/local/share/cmake-3.11/Modules/FindASPELL.cmake
      /opt/local/share/cmake-3.11/Modules/FindAVIFile.cmake
      /opt/local/share/cmake-3.11/Modules/FindArmadillo.cmake


CMake RPATH behaviour change
--------------------------------

In opticks/examples/UseBoost/CMakeLists.txt With::

    cmake_minimum_required(VERSION 3.5 FATAL_ERROR)  ## same with 3.4 3.3  (3.2 has other issues)
    ...
    set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

Get error about duplicated path that doesnt appear with minimum_required of 3.1::

    -- Up-to-date: /usr/local/opticks-cmake-overhaul/lib/libUseBoost.dylib
    error: /opt/local/bin/install_name_tool: for: /usr/local/opticks-cmake-overhaul/lib/libUseBoost.dylib (for architecture x86_64) option "-add_rpath /usr/local/opticks-cmake-overhaul/lib" would duplicate path, file already has LC_RPATH for: /usr/local/opticks-cmake-overhaul/lib
    -- Installing: /usr/local/opticks-cmake-overhaul/lib/pkgconfig/useboost.pc



cmake_minimum_required effects
----------------------------------

::

     01 cmake_minimum_required(VERSION 2.6)
      2 
      3 set(name UseGLM)
      4 
      5 #  cmake --help-policy CMP0048
      6 #if(POLICY CMP0048)
      7 #   cmake_policy(SET CMP0048 NEW) # CMake 3.0.0
      8 #endif()
      9 
     10 project(${name} VERSION 0.1.0)
     11 


Support for the version in the project command only started with 3.0, so the 
above gives::

    CMake Error at CMakeLists.txt:10 (project):
      VERSION not allowed unless CMP0048 is set to NEW

Increasing cmake_minimum_required to 3.0 avoids the error.  Also staying with 2.6 
and uncommenting cmake_policy setting does the same.



find_package search path 
---------------------------

* https://cmake.org/cmake/help/v3.0/command/find_package.html

1. First "MODULE" mode looks for Find<Package>.cmake in CMAKE_MODULE_PATH and then the CMake installation
   (so if you want to avoid catching a standard package ensure the name is prefixed by something like Opticks)
2. Second "CONFIG" mode searches for a file called <name>Config.cmake or <lower-case-name>-config.cmake


CMake versioning
-------------------

::

# https://stackoverflow.com/questions/35300833/cmake-get-output-of-some-command-every-build-and-reconfigure-files-that-depends

#set(VERSIONING_FILES src/versioning.cmake SVersion.cc.in)
#
#find_program(HG hg DOC "Mercurial executable file")
#
#add_custom_target(versioning ALL COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_SOURCE_DIR}/SVersion.cpp.in
#    SOURCES ${VERSIONING_FILES})
#
#add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/src/version.cpp
#    COMMAND ${CMAKE_COMMAND} -DVERSION_INPUT=${CMAKE_SOURCE_DIR}/src/version.cpp.in -DVERSION_OUTPUT=${CMAKE_SOURCE_DIR}/src/version.cpp -DHG=${HG} -P ${CMAKE_SOURCE_DIR}/src/versioning.cmake
#    DEPENDS ${CMAKE_SOURCE_DIR}/src/version.cpp.in)
#..
#
#if (HG)
#    execute_process(COMMAND ${HG} id OUTPUT_VARIABLE REVISION OUTPUT_STRIP_TRAILING_WHITESPACE)
#else ()
#    set(REVISION "<undefined>")
#endif ()
#configure_file(${VERSION_INPUT} ${VERSION_OUTPUT})
#execute_process(COMMAND ${CMAKE_COMMAND} -E touch ${VERSION_OUTPUT})    # invalidate output to force rebuild this source file even if it was not changed.
#




Help
------

::

    epsilon:util blyth$ cmake --help-module FindBoost
    FindBoost
    ---------

    Find Boost include dirs and libraries

    ...



    epsilon:opticks blyth$ cmake -DCMAKE_MODULE_PATH=./cmake/Modules --help-module FindGLM
    Argument "FindGLM" to --help-module is not a CMake module.
    epsilon:opticks blyth$ 



CMake packages/modules/config/...
-----------------------------------

* https://cmake.org/cmake/help/v3.3/manual/cmake-packages.7.html#manual:cmake-packages(7)

Export/Import of targets
--------------------------

* https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/Exporting-and-Importing-Targets


Learning from BCM (Boost CMake Modules) project
---------------------------------------------------

* Looks promising way to cut down on the boilerplate, see bcm-

* http://bcm.readthedocs.io/en/latest/src/Building.html

  Very clear explanation describing a standalone CMake setup for building boost_filesystem


string REPLACE
-----------------

::

    string(REPLACE <match_string>
           <replace_string> <output variable>
           <input> [<input>...])


cmake generator expressions
------------------------------

* https://cmake.org/cmake/help/v3.3/manual/cmake-generator-expressions.7.html

::

    $<0:...>
        Empty string (ignores ...)

    $<1:...>
        Content of ...

    $<INSTALL_INTERFACE:...>
        Content of ... when the property is exported using install(EXPORT), and empty otherwise.

    $<BUILD_INTERFACE:...>
        Content of ... when the property is exported using export(), 
        or when the target is used by another target in the same buildsystem. 
        Expands to the empty string otherwise.

    $<INSTALL_PREFIX>
         Content of the install prefix when the target is exported via install(EXPORT) and empty otherwise.

    $<TARGET_PROPERTY:tgt,prop>
        Value of the property prop on the target tgt.
        Note that tgt is not added as a dependency of the target this expression is evaluated on.



Uses "$<0:...>" gen expression acts to skip the build interface::

     75 function(bcm_preprocess_pkgconfig_property VAR TARGET PROP)
     76     get_target_property(OUT_PROP ${TARGET} ${PROP})
     77     string(REPLACE "$<BUILD_INTERFACE:" "$<0:" OUT_PROP "${OUT_PROP}")
     78     string(REPLACE "$<INSTALL_INTERFACE:" "$<1:" OUT_PROP "${OUT_PROP}")
     79 
     80     string(REPLACE "$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}" "\${includedir}" OUT_PROP "${OUT_PROP}")
     81     string(REPLACE "$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}" "\${libdir}" OUT_PROP "${OUT_PROP}")
     82     string(REPLACE "$<INSTALL_PREFIX>" "\${prefix}" OUT_PROP "${OUT_PROP}")
     83 
     84     set(${VAR} ${OUT_PROP} PARENT_SCOPE)
     85 
     86 endfunction()




include(GNUInstallDirs)
--------------------------

* https://cmake.org/cmake/help/v3.0/module/GNUInstallDirs.html

Define GNU standard installation directories, eg::

   CMAKE_INSTALL_BINDIR
   CMAKE_INSTALL_LIBDIR
   CMAKE_INSTALL_INCLUDEDIR    ## relative to CMAKE_INSTALL_PREFIX
  
   CMAKE_INSTALL_FULL_INCLUDEDIR   ## absolute, including the CMAKE_INSTALL_PREFIX



Effective CMake Daniel Pfeifer
---------------------------------

* https://github.com/boostcon/cppnow_presentations_2017/blob/master/05-19-2017_friday/effective_cmake__daniel_pfeifer__cppnow_05-19-2017.pdf

* https://www.reddit.com/r/cpp/comments/6fv0bh/cnow_2017_daniel_pfeifer_effective_cmake_pdf_in/

  video from YouTube

* cmak-effective

p30
~~~~

* Non-INTERFACE_ properties define the build specification of a target.
* INTERFACE_ properties define the usage requirements of a target.

p31
~~~~

* PRIVATE populates the non-INTERFACE_ property. 
* INTERFACE populates the INTERFACE_ property.
* PUBLIC populates both.

p36
~~~~

::

   add_library(Bar INTERFACE)
   target_compile_definitions(Bar INTERFACE BAR=1)

* INTERFACE libraries have no build specification. 
* They only have usage requirements.



It's Time To Do CMake Right
----------------------------

* https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/
* https://github.com/pabloariasal/modern-cmake-sample

Target properties are defined in one of two scopes: INTERFACE and PRIVATE.
Private properties are used internally to build the target, while interface
properties are used externally by users of the target. In other words,
interface properties model usage requirements, whereas private properties model
build requirements of targets.



find_dependency
-----------------

* https://cmake.org/cmake/help/v3.2/module/CMakeFindDependencyMacro.html


PUBLIC/INTERFACE/PRIVATE
---------------------------

* https://stackoverflow.com/questions/26037954/cmake-target-link-libraries-interface-dependencies


If you are creating a shared library and your source cpp files #include the
headers of another library (Say, QtNetwork for example), but your header files
don't include QtNetwork headers, then QtNetwork is a PRIVATE dependency.

If your source files and your headers include the headers of another library,
then it is a PUBLIC dependency.

If your header files but not your source files include the headers of another
library, then it is an INTERFACE dependency.

Other build properties of PUBLIC and INTERFACE dependencies are propagated to
consuming libraries.
http://www.cmake.org/cmake/help/v3.0/manual/cmake-buildsystem.7.html#transitive-usage-requirements


generator expressions
-------------------------

* https://cmake.org/cmake/help/v3.0/manual/cmake-generator-expressions.7.html

::

    $<INSTALL_INTERFACE:...>
    Content of ... when the property is exported using install(EXPORT), 
    and empty otherwise.

    $<BUILD_INTERFACE:...>
    Content of ... when the property is exported using export(), or when the
    target is used by another target in the same buildsystem. Expands to the empty
    string otherwise.


* http://cmake.3232098.n2.nabble.com/Boost-s-CMAKE-approach-and-the-BUILD-INTERFACE-generator-expression-td7596498.html

On Fri, 2017-10-27 at 12:22 -0700, Wesley Smith wrote:

> Boost's CMAKE page (http://bcm.readthedocs.io/en/latest/src/Building.html)  
> says: 
> 
> So this will build the library named boost_filesystem, however, we need to 
> supply the dependencies to boost_filesystem and add the include directories. 
> To add the include directory we use target_include_directories. For this, we 
> tell cmake to use local include directory, but since this is only valid 
> during build and not after installation, we use the BUILD_INTERFACE 
> generator expression so that cmake will only use it during build and not 
> installation: 
> 
> target_include_directories(boost_filesystem PUBLIC 
>     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
> ) 
> 
> 
> Is is necessary to use a BUILD_INTERFACE here?  Couldn't you use 
> PUBLIC/PRIVATE/INTERFACE to achieve the same effect?  What are the use cases 
> over for BUILD_INTERFACE that setting include dirs as 
> PUBLIC/PRIVATE/INTERFACE doesn't cover?
«  [hide part of quote]

You don't need `BUILD_INTERFACE` if it is set to `PRIVATE`, as none of the 
downstream users will use the include. However, when using `PUBLIC` or 
`INTERFACE` you will need `BUILD_INTERFACE`, and in the example above it using 
`PUBLIC`. 

This is because there are two types of consumers using the target. One is a 
target within the build. This will use the include directory from the 
source(or build) directory. The `BUILD_INTERFACE` ensures that this is only 
used for this type of consumer. 

The other type of consumer is the what is used after installation. In this 
case the include directory is different and most likely points to the 
directory in installation directory like `${CMAKE_INSTALL_PREFIX}/include`. 
The `INSTALL_INTERFACE` ensures that this is only used for this type of 
consumer. 

So for this case, you setup the includes something like this: 

target_include_directories(boost_filesystem PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include> 
) 

This will ensure that each type of consumer get the correct include directory. 
Ultimately, you don't want the user to add an include to a local source 
directory, as this could have surprising side effect. Fortunately, cmake 
ensures that this won't happen either, by producing an error if the 
installations include paths that point to directories in the source or build 
directory. 

At the same token, you also don't want the build to point to the installation 
include as well as this may include headers you don't want to use. 

Finally, in the example above it didn't use the `INSTALL_PREFIX`. This is 
because the install sets it correctly when using the `INCLUDES DESTINATION` 
statement: 

install(TARGETS boost_filesystem EXPORT boost_filesystem-targets 
    RUNTIME DESTINATION bin 
    LIBRARY DESTINATION lib 
    ARCHIVE DESTINATION lib 
    INCLUDES DESTINATION include 
) 

Paul 
-- 


transitive usage requirements
--------------------------------

* https://cmake.org/cmake/help/v3.0/manual/cmake-buildsystem.7.html#transitive-usage-requirements



ClimbingStats/UpstreamLib example
-----------------------------------

* https://cmake.org/cmake/help/v3.0/manual/cmake-packages.7.html

::

    -install(TARGETS ${name}  DESTINATION lib)
    +install(TARGETS ${name}  EXPORT ${name}Targets 
    +        LIBRARY DESTINATION lib 
    +        INCLUDES DESTINATION include/${name}
    +)
     install(FILES ${HEADERS} DESTINATION include/${name})
    +install(EXPORT ${name}Targets   DESTINATION target)




FindBoost example
-------------------

* /opt/local/share/cmake-3.11/Modules/FindBoost.cmake

 


How to CMake "vend" like G4 ?
----------------------------------

/usr/local/opticks/externals/g4/geant4_10_04_p01/examples/README.HowToRun::

 31     1a) With CMake
 32 
 33         % cd path_to_exampleXYZ     # go to directory which contains your example
 34         % mkdir exampleXYZ_build
 35         % cd exampleXYZ_build
 36         % cmake -DGeant4_DIR=path_to_Geant4_installation/lib[64]/Geant4-10.0.0/ ../exampleXYZ
 37         % make -j N exampleXYZ      # "N" is the number of processes
 38         % make install              # this step is optional


::

    epsilon:~ blyth$ ll /usr/local/opticks/externals/lib/Geant4-10.2.1/
    total 144
    -rw-r--r--   1 blyth  staff   3166 Mar  2  2016 UseGeant4.cmake
    -rw-r--r--   1 blyth  staff   1050 Apr  4 14:32 Geant4ConfigVersion.cmake
    -rw-r--r--   1 blyth  staff  26222 Apr  4 14:32 Geant4Config.cmake
    -rw-r--r--   1 blyth  staff  18092 Apr  4 14:32 Geant4LibraryDepends.cmake
    -rw-r--r--   1 blyth  staff  14853 Apr  4 14:32 Geant4LibraryDepends-relwithdebinfo.cmake
    lrwxr-xr-x   1 blyth  staff      2 Apr  4 15:20 Darwin-clang -> ..
    drwxr-xr-x   9 blyth  staff    288 Apr  4 15:20 .
    drwxr-xr-x  11 blyth  staff    352 Apr  4 15:20 Modules
    drwxr-xr-x  55 blyth  staff   1760 Apr  4 15:20 ..
    epsilon:~ blyth$ 


* https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html

::

   /usr/local/opticks/build/cmake_install.cmake





EOU
}


#cmak-base(){ echo $(local-base)/env/tools ; }
cmak-base(){ echo /tmp/$USER/env/tools ; }
cmak-dir(){  echo $(cmak-base)/cmak ; }
cmak-bdir(){ echo $(cmak-base)/cmak/build ; }


cmak-cd(){
    local dir=$(cmak-dir)
    mkdir -p $dir
    cd $dir  
}

cmak-bcd(){
    local bdir=$(cmak-bdir)
    mkdir -p $bdir
    cd $bdir  
}

cmak-brm(){
    local bdir=$(cmak-bdir)
    rm -rf $bdir
}



cmak-txt-dev-(){ cat << EOD


set(Boost_DEBUG 1)
set(Boost_USE_STATIC_LIBS 1)
set(Boost_NO_SYSTEM_PATHS 1)

set(CMAKE_MODULE_PATH \$ENV{OPTICKS_HOME}/cmake/Modules)
set(OPTICKS_PREFIX "\$ENV{LOCAL_BASE}/opticks")
set(BOOST_LIBRARYDIR \$ENV{LOCAL_BASE}/opticks/externals/lib)
set(BOOST_INCLUDEDIR \$ENV{LOCAL_BASE}/opticks/externals/include)
message(" OPTICKS_PREFIX  : \${OPTICKS_PREFIX} ")

EOD
}



cmak-vars-(){ local name=${1:-CMAKE} ; cat << EOV
${name}_CXX_FLAGS
${name}_CXX_FLAGS_DEBUG
${name}_CXX_FLAGS_MINSIZEREL
${name}_CXX_FLAGS_RELEASE
${name}_CXX_FLAGS_RELWITHDEBINFO
${name}_EXE_LINKER_FLAGS
${name}_PREFIX_PATH
${name}_SYSTEM_PREFIX_PATH
${name}_INCLUDE_PATH
${name}_LIBRARY_PATH
${name}_PROGRAM_PATH
EOV
}


cmak-vars(){
   local name=${1:-$FUNCNAME} 
   local var
   cmak-vars- CMAKE | while read var 
   do 
      cat << EOV
message("\${name}.$var : \${$var} ")
EOV
   done

}


cmak-package-vars-(){ local name=${1:-Geant4} ; cat << EOV
${name}_LIBRARY
${name}_LIBRARIES
${name}_INCLUDE_DIRS
${name}_DEFINITIONS
EOV
}
cmak-package-vars(){
   local name=${1:-OpenMesh} 
   local var
   cmak-package-vars- $name | while read var 
   do 
      cat << EOV
message("\${name}.$var : \${$var} ")
EOV
   done
}





cmak-txt-(){
     local name=$1
     cat << EOH
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(tt)

EOH
     cmak-vars 

     local find=NO
     [ "$find" == "YES" ] && cat << EOF
find_package($*)

message("${name}_LIBRARY       : \${${name}_LIBRARY}")
message("${name}_LIBRARIES     : \${${name}_LIBRARIES}")
message("${name}_INCLUDE_DIRS  : \${${name}_INCLUDE_DIR}")
message("${name}_DEFINITIONS   : \${${name}_DEFINITIONS}")

EOF

     case $name in
        Boost) cmak-txt-qwns- $name ;;
     esac  
}

cmak-txt-qwns-(){
   local name=$1
   for qwn in $(cmak-qwns-$name)   
   do cat << EOQ
message("$qwn  : \${$qwn}")
EOQ
   done
}



cmak-qwns-Boost(){ cat << EOV
_Boost_MISSING_COMPONENTS
_boost_DEBUG_NAMES
Boost_FIND_COMPONENTS
Boost_NUM_COMPONENTS_WANTED
Boost_NUM_MISSING_COMPONENTS
BOOST_ROOT
BOOST_LIBRARYDIR
Boost_LIB_PREFIX
_boost_RELEASE_NAMES
EOV
}


cmak-find-boost(){

   type $FUNCNAME

   cmak-cd
   cmak-txt- Boost REQUIRED COMPONENTS system thread program_options log log_setup filesystem regex > CMakeLists.txt
   cat CMakeLists.txt

   cmak-brm
   cmak-bcd

   boost-
   local prefix=$(boost-prefix)

   # [ "${prefix::3}" == "/c/" ] && prefix=C:${prefix:2}   
   # [ "${prefix::3}" == "/c/" ] && prefix=/$prefix
   #
   #
   # file:///C:/Program%20Files/Git/ReleaseNotes.html
   # 
   #    git-for-windows gitbash path conversion kicks in for paths starting with a slash
   #    to avoid the POSIX-to-Windows path convertion either 
   #    temporarily set MSYS_NO_PATHCONV or use double slash 
   #

   local src=$(cmak-dir)

   #MSYS_NO_PATHCONV=1 
   cmake $src \
           -DBOOST_ROOT=$prefix \
           -DBoost_NO_SYSTEM_PATHS=ON \
           -DBoost_USE_STATIC_LIBS=ON

   #       -DBOOST_LIBRARYDIR=$prefix/lib  \
   #       -DBOOST_INCLUDEDIR=$prefix/include
   #       -DBOOST_ROOT=$prefix
   #
   #
   #   twas failing to find libs due to a lib suffix
   #   switched that on using the Boost_USE_STATIC_LIBS switch
   #

}



cmak-flags(){

   cmak-cd
   cmak-txt- > CMakeLists.txt
   cat CMakeLists.txt

   cmak-brm
   cmak-bcd

   local src=$(cmak-dir)
   cmake $src 

}


cmak-opticks-txt-(){
     local name=$1
     cat << EOH

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
   message(FATAL_ERROR "in-source build detected : DONT DO THAT")
endif()

set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX \$ENV{OPTICKS_HOME}/cmake/Modules/Geant4MakeRules_cxx.cmake)

set(name CMakOpticksTxt)
project(TestFinding${name})

set(CMAKE_MODULE_PATH "\$ENV{OPTICKS_HOME}/cmake/Modules")

include(EnvBuildOptions)


EOH
     cmak-vars 

     local find=YES
     [ "$find" == "YES" ] && cat << EOF
find_package($*)

EOF
}



cmak-find-pkg(){

   local pkg=${1:-OpenMesh} 
   shift

   local iwd=$PWD

   cmak-cd
   cmak-opticks-txt- $pkg > CMakeLists.txt
   cmak-package-vars $pkg >> CMakeLists.txt 

   cat CMakeLists.txt

   cmak-brm
   cmak-bcd
   local src=$(cmak-dir)

   cmake $src $*

   cd $iwd
}


cmak-find-OpticksGLEW(){     cmak-find-pkg OpticksGLEW $* ; }
cmak-find-OpticksGLFW(){     cmak-find-pkg OpticksGLFW $* ; }
cmak-find-Opticks(){         cmak-find-pkg Opticks $* ; }
cmak-find-OpenMesh(){ cmak-find-pkg OpenMesh $* ; }





cmak-stem(){
    local filename=$1 
    local stem 
    case $filename in 
        *.cc) echo ${filename/.cc}  ;;
       *.cpp) echo ${filename/.cpp} ;;  
           *) echo ERROR ;;
    esac
}


cmak-cc-(){ 

    local filename=$1
    local stem=$(cmak-stem $filename)

    cat << EOS
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(name $stem)

project(\${name})

find_package(Boost REQUIRED COMPONENTS date_time)

add_executable(\${name} $filename)

target_include_directories(\${name} PUBLIC \${Boost_INCLUDE_DIRS} )

target_link_libraries(\${name} \${Boost_LIBRARIES} )


EOS
}


cmak-config(){ echo Debug ; }
cmak-cc(){

   local filename=$1
   local stem=$(cmak-stem $filename)
   local iwd=$PWD
  
   cmak-cd

   cp $iwd/$filename .
   cmak-cc- $filename > CMakeLists.txt
   cat CMakeLists.txt

   cmak-brm
   cmak-bcd


   boost- 

   cmake \
           -DBOOST_ROOT=$(boost-prefix) \
           -DBoost_NO_SYSTEM_PATHS=ON \
           -DBoost_USE_STATIC_LIBS=ON \
           ..



   cmake --build . --config $(cmak-config) --target ALL_BUILD

   cd $iwd
}

cmak-bin()
{
   echo $(cmak-bdir)/$(cmak-config)/$1.exe
}


cmak-check-threads-(){ cat << EOT
cmake_minimum_required(VERSION 2.8.6)
#FIND_PACKAGE (Threads)

EOT
}

cmak-check-threads()
{
   local iwd=$PWD
   local tmp=/tmp/env/$FUNCNAME
   rm -rf $tmp
   mkdir -p $tmp

   cd $tmp
   cmak-check-threads- > CMakeLists.txt 

   mkdir build
   cd build

   cmake ..   


   cd $iwd
}

cmak-effective(){ open ~/opticks_refs/effective_cmake__daniel_pfeifer__cppnow_05-19-2017.pdf  ; }
