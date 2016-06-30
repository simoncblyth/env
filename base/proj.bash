proj-rel(){      echo base ; }
proj-src(){      echo $(proj-rel)/proj.bash ; }
proj-source(){   echo ${BASH_SOURCE:-$(env-home)/$(proj-src)} ; }
proj-vi(){       vi $(proj-source) $(proj-sdir)/CMakeLists.txt $(proj-tdir)/CMakeLists.txt ; }
proj-usage(){ cat << EOU


EOU
}

proj-env(){  
   elocal- 
   opticks-
}

proj-dir(){  echo $(env-home)/$(proj-rel) ; }
proj-sdir(){ echo $(env-home)/$(proj-rel) ; }
proj-tdir(){ echo $(env-home)/$(proj-rel)/tests ; }
proj-idir(){ echo $(local-base)/env/$(proj-rel).install ; }
proj-bdir(){ echo $(local-base)/env/$(proj-rel).build ; }
proj-prefix(){ echo $(proj-idir) ; }


proj-cd(){   cd $(proj-dir); }
proj-scd(){  cd $(proj-sdir); }
proj-tcd(){  cd $(proj-tdir); }
proj-bcd(){  cd $(proj-bdir); }
proj-icd(){  cd $(proj-idir); }

proj-name(){ echo Proj ; }
proj-tag(){  echo PROJ; }

proj-wipe(){ local bdir=$(proj-bdir) ; rm -rf $bdir ; } 
proj-txt(){ vi $(proj-sdir)/CMakeLists.txt $(proj-tdir)/CMakeLists.txt ; } 

proj-cmake(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD
   local bdir=$(proj-bdir)
   mkdir -p $bdir

   [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already use proj-cmake-modify to change config or proj-configure to start from scratch  && return  
   proj-bcd

   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(proj-prefix) \
       $* \
       $(proj-sdir)

   cd $iwd 
}


proj-cmake-modify(){
  local msg="=== $FUNCNAME : "
  local bdir=$(proj-bdir)
  local bcache=$bdir/CMakeCache.txt
  [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return
  proj-bcd

  cmake \
          . 
}

proj-configure(){
   proj-wipe
   proj-cmake
}
proj-all(){
   proj-configure
   proj--
   proj-ctest
}

proj-sln(){ echo $(proj-bdir)/$(proj-name).sln ; }
proj-slnw(){  vs- ; echo $(vs-wp $(proj-sln)) ; }
proj-vs(){
   vs-
   local sln=$1
   [ -z "$sln" ] && sln=$(proj-sln)
   local slnw=$(vs-wp $sln)

    cat << EOC
# sln  $sln
# slnw $slnw
# copy/paste into powershell v2 
#
devenv /useenv $slnw
EOC

}

#proj-config(){ echo Debug ; }
proj-config(){ echo RelWithDebInfo ; }
proj--(){

   local msg="$FUNCNAME : "
   local iwd=$PWD

   local bdir=$(proj-bdir)
   [ ! -d "$bdir" ] && echo $msg bdir $bdir does not exist && return

   cd $bdir

   cmake --build . --config $(proj-config) --target ${1:-install}

   cd $iwd
}

proj-ctest()
{
   local msg="$FUNCNAME : "
   local iwd=$PWD

   local bdir=$1
   shift
   [ -z "$bdir" ] && bdir=$(proj-bdir)
   cd $bdir

   ctest $*

   cd $iwd
   echo $msg use -V to show output 
}




proj-end-template(){ echo 130 ; }
## NB KEEP THE LINE NUMBER RETURNED BY proj-end-template updated to define the template

proj-notes(){
   cat << EON


     env-;env-fgenproj graphics/openmeshclient/omc 


     proj-;proj-gen- graphics/openmeshclient/omc
          emit to stdout the filled out template for a new bash proj   

     proj-gen xml/xmldiff
          save the proj-gen- output to $(env-home)/xml/xmldiff.bash 
          if no such file exists 

     proj-gen base/hello <repo>
          generate $(<repo>-home)/base/hello/hello.bash 
          hook up the precursor into $(<repo>-home)/<repo>.bash
          eval the precursor


     proj-isfunc-  name
          detect if the function "name" is defined 

          proj-isfunc envx-
          n
          proj-isfunc env-
          y
EON
}


proj-isfunc-(){ local n=$1 ; [ "$(type $n 2>/dev/null | head -1 )" == "$n is a function" ] && return 0 || return 1 ; }
proj-isfunc(){ $FUNCNAME- $* && echo y || echo n ;  }




proj-gen-repo(){
  echo ${1:-env}
}

proj-gen-rel(){
  local path=${2:-dummy/hello}
  local rel=$(dirname $path)
  echo $rel
}
proj-gen-path(){
  local path=${2:-dummy/hello}
  local rel=$(dirname $path)
  local name=$(proj-gen-name $*)
  echo $rel/$name.bash
}
proj-gen-heading(){
   local path=${2:-dummy/hello}
   local dir=$(dirname $path)
   echo $dir   
}
proj-gen-name(){
  local name=$(basename ${2:-dummy/hello})  
  name=${name/.bash}
  echo $name  
}
proj-gen-pkg(){
  echo ${3:-Package}
}




proj-gen-(){

  local msg="=== $FUNCNAME :"
  local fgd=$(proj-gen-rel $*)
  local fgp=$(proj-gen-path $*)
  local fgn=$(proj-gen-name $*)
  local fgr=$(proj-gen-repo $*)
  local fgh=$(proj-gen-heading $*)
  local pkg=$(proj-gen-pkg $*)

  echo \# $msg $* fgd $fgd fgp $fgp fgn $fgn fgh $fgh pkg $pkg

  head -$(proj-end-template) $(proj-source) \
         | perl -p -e "s,$(proj-src),$fgp," - \
         | perl -p -e "s, base , $fgd ,g" - \
         | perl -p -e "s,proj,$fgn,g" - \
         | perl -p -e "s,env-home,$fgr-home,g" - \
         | perl -p -e "s,/env,/$fgr,g" - \
         | perl -p -e "s,heading,$fgh,g" - \
         | cat 

}

proj-precursor-(){
  local fgp=$(proj-gen-path $*)
  local fgn=$(proj-gen-name $*)
  local fgr=$(proj-gen-repo $*)
cat << EOP
$fgn-(){      . \$($fgr-home)/$fgp && $fgn-env \$* ; }
EOP
}

proj-cmak-(){
     local name=$1
     shift

     cat << \EOH
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(name %NAME%)
project(${name})


set(CMAKE_MODULE_PATH "$ENV{ENV_HOME}/cmake/Modules")

include(EnvBuildOptions)  # RPATH setup
include(CTest)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})


set(pkg %PKG%)
find_package(${pkg})

message("${pkg}_LIBRARY       : ${${pkg}_LIBRARY}")
message("${pkg}_LIBRARIES     : ${${pkg}_LIBRARIES}")
message("${pkg}_INCLUDE_DIRS  : ${${pkg}_INCLUDE_DIRS}")
message("${pkg}_DEFINITIONS   : ${${pkg}_DEFINITIONS}")

set(LIBRARIES
   ${${pkg}_LIBRARIES}
)

message("${name}:LIBRARIES : ${LIBRARIES} ")

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}
   ${${pkg}_INCLUDE_DIRS} 
)

add_definitions(
    ${${pkg}_DEFINITIONS} 
)

set(SOURCES
    Args.cc
) 

set(HEADERS
    Args.hh
)

add_library( ${name}  SHARED ${SOURCES})
target_link_libraries( ${name}  ${LIBRARIES} )

install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


EOH
}

proj-cmak-tests-(){
     local name=$1
     shift

     cat << \EOT

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(name %NAME%Test)
project(${name})

set(TEST_SOURCES
    ArgsTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ${LIBRARIES} %NAME%)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

EOT
}


proj-args-hh-(){ local tag=$1 ; cat << EOH
#pragma once

#include "${tag}_API_EXPORT.hh"

struct ${tag}_API Args {
   int    argc ; 
   char** argv ;
    
   Args(int argc_, char** argv_);
   void  Summary(const char* msg="Args::Summary") ;
};

EOH
}

proj-args-cc-(){ cat << EOX
#include <iostream>
#include "Args.hh"
   
Args::Args(int argc_, char** argv_) 
     :
     argc(argc_),
     argv(argv_)
{
}
void Args::Summary(const char* msg)
{
    std::cerr << msg << std::endl ;
    for(int i=0 ; i < argc ; i++)
       std::cerr << argv[i] << std::endl ; 

}
EOX
}

proj-argstest-cc-(){ cat << EOX

#include "Args.hh"

int main(int argc, char** argv)
{
    Args a(argc, argv);
    a.Summary();
    return 0 ; 
}

EOX
}


proj-api-export-hh-(){ local tag=$1 ; local proj=$2 ; cat << EOX

#pragma once

#if defined (_WIN32) 

   #if defined(${proj}_EXPORTS)
       #define  ${tag}_API __declspec(dllexport)
   #else
       #define  ${tag}_API __declspec(dllimport)
   #endif

#else

   #define ${tag}_API  __attribute__ ((visibility ("default")))

#endif

EOX
}




proj-fgen(){
  local msg="=== $FUNCNAME :"

  local fgr=$(proj-gen-repo $*)
  local fgp=$(proj-gen-path $*)
  local path=$($fgr-home)/$fgp  
  echo $msg fgr $fgr fgp $fgp path $path args $*
  rm $path

  proj-gen $*
}


proj-gen(){

  local msg="=== $FUNCNAME :"

  local fgd=$(proj-gen-rel  $*)
  local fgp=$(proj-gen-path $*)
  local fgn=$(proj-gen-name $*)
  local fgr=$(proj-gen-repo $*)
  local pkg=$(proj-gen-pkg $*)

  local proj=$fgn 
  local tag=$(echo $proj | tr "[a-z]" "[A-Z]" ) 

  echo  $msg  .... fgp:$fgp fgn:$fgn fgr:$fgr pkg $pkg tag $tag

  local path=$($fgr-home)/$fgp  
  local dir=$(dirname $path)
  local top=$($fgr-home)/$fgr.bash

  [ ! -f "$top" ] && echo $msg the repo $repo must have a top .bash at $top && return 1
  [ -f "$path" ]  && echo $msg ABORT : path $path exists already ... delete and rerun to override && return 0  
  
  echo;echo $msg proposes to write the below into : $path ;echo
  proj-gen- $*

  echo;echo $msg and hookup precursor into : $top;echo 
  proj-precursor- $*
  echo

  local ans
  read -p "$msg enter YES to proceed : " ans
  [ "$ans" != "YES" ] && echo $msg skipping && return 0 
 
  [ ! -d "$dir" ] && echo $msg WARNING : creating dir $dir &&  mkdir -p "$dir" 
  proj-gen- $* > $path

  echo APPENDING PRECURSOR

  proj-precursor- $* >> $top

  local tdir=$dir/tests 
  [ ! -d "$tdir" ] && echo $msg WARNING : creating tdir $tdir && mkdir -p "$tdir" 

  proj-cmak-       | perl -p -e "s,%NAME%,$fgn," - \
                   | perl -p -e "s,%PKG%,$pkg,"  > $dir/CMakeLists.txt

  proj-cmak-tests- | perl -p -e "s,%NAME%,$fgn," -  > $tdir/CMakeLists.txt




  proj-args-hh- $tag > $dir/Args.hh
  proj-args-cc- > $dir/Args.cc
  proj-argstest-cc- > $tdir/ArgsTest.cc

  proj-api-export-hh-  $tag $proj > $dir/${proj}_API_EXPORT.hh 



  echo $msg defining precursor $fgn- 
  eval $(proj-precursor- $*)

  echo $msg invoking precursor $fgn-
  eval $fgn-
  eval $fgn-vi

}



