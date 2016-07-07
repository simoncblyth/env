cmak-src(){      echo tools/cmak.bash ; }
cmak-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cmak-src)} ; }
cmak-vi(){       vi $(cmak-source) ; }
cmak-env(){      elocal- ; }
cmak-usage(){ cat << EOU





EOU
}
cmak-dir(){ echo $(local-base)/env/tools/cmak ; }
cmak-bdir(){ echo $(local-base)/env/tools/cmak/build ; }
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
project(${name})

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


cmak-find-GLEW(){     cmak-find-pkg GLEW $* ; }
cmak-find-GLFW(){     cmak-find-pkg GLFW $* ; }
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


