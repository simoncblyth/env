# === func-gen- : numpy/cnpy fgp numpy/cnpy.bash fgn cnpy fgh numpy
cnpy-src(){      echo numpy/cnpy.bash ; }
cnpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cnpy-src)} ; }
cnpy-vi(){       vi $(cnpy-source) ; }
cnpy-env(){      elocal- ; }
cnpy-usage(){ cat << EOU

CNPY
=====

* https://github.com/rogersce/cnpy.git

dstalke fork 
------------

The fork has some improvements but rapidly starts
using c++11 stuff, which I expect will cause 
portability issues, so sticking with original, 
may be able to grab some of developments avoiding c++11 stuff

* https://github.com/dstahlke/cnpy

Added as NuWa Utility
-----------------------

Could be an external, but I suspect I will 
want to make extensive changes if using more
extensively.  So added to Utilities



Issues
-------




CMake first build issue
~~~~~~~~~~~~~~~~~~~~~~~~

Fails to build the example until 2nd run, as
needs lib to be installed

RPATH not working 
~~~~~~~~~~~~~~~~~~

Added RPATH setup to CMakeLists, see cnpytest- which 
takes advantage of this

Have to use git urls not https ones on N
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Belle7 lives in a wierd network environment, or
maybe just a very old git ?

::

    [blyth@belle7 dybgaudi]$ cnpy-get
    Initialized empty Git repository in /data1/env/local/env/numpy/cnpy/.git/
    Cannot get remote repository information.
    Perhaps git-update-server-info needs to be run there?
    [blyth@belle7 numpy]$ 
    [blyth@belle7 numpy]$ 
    [blyth@belle7 numpy]$ which git 
    /usr/bin/git






CMakeLists Mods
------------------

Install header into containing cnpy folder and 
RPATH setup.


::

    (chroma_env)delta:cnpy blyth$ git diff CMakeLists.txt 
    diff --git a/CMakeLists.txt b/CMakeLists.txt
    index 5a7cdd3..5748e8d 100644
    --- a/CMakeLists.txt
    +++ b/CMakeLists.txt
    @@ -3,6 +3,10 @@ if(COMMAND cmake_policy)
            cmake_policy(SET CMP0003 NEW)
     endif(COMMAND cmake_policy)
     
    +set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}  "$ENV{ENV_HOME}/cmake/Modules")
    +include(EnvBuildOptions)
    +
    +
     project(CNPY)
     
     option(ENABLE_STATIC "Build static (.a) library" ON)
    @@ -17,7 +21,7 @@ if(ENABLE_STATIC)
         install(TARGETS "cnpy-static" ARCHIVE DESTINATION lib)
     endif(ENABLE_STATIC)
     
    -install(FILES "cnpy.h" DESTINATION include)
    +install(FILES "cnpy.h" DESTINATION include/cnpy)
     install(FILES "mat2npz" "npy2mat" "npz2mat" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
     
     add_executable(example1 example1.cpp)






EOU
}
cnpy-prefix(){ echo $(local-base)/env/cnpy ; } 
cnpy-sdir(){ echo $(local-base)/env/numpy/cnpy ; }
cnpy-scd(){  cd $(cnpy-sdir); }

cnpy-url(){
   case $1 in 
     rogersce)    echo git://github.com/rogersce/cnpy.git  ;;
     simoncblyth) echo git://github.com/simoncblyth/cnpy.git ;;
     dstalke)     echo git://github.com/dstahlke/cnpy.git ;; 
   esac
}
cnpy-vers(){ echo rogersce ; }

cnpy-get(){
   local dir=$(dirname $(cnpy-sdir)) &&  mkdir -p $dir && cd $dir
   [ ! -d cnpy ] &&  git clone $(cnpy-url $(cnpy-vers)) 
}

cnpy-wipe(){ rm -rf $(cnpy-sdir) ; }

cnpy-build(){

   cnpy-scd

   mkdir build
   cd build

   cmake -DCMAKE_INSTALL_PREFIX=$(cnpy-prefix) ..

   export VERBOSE=1

   make 
   make install

   cd ..
   #rm -rf build


}


cnpy-names(){ cat << EON
LICENSE
README
cnpy.h
cnpy.cpp
EON
}

cnpy-mapping(){
  case $1 in 
    LICENSE|README) echo $1 ;;
            cnpy.h) echo cnpy/$1 ;;
          cnpy.cpp) echo src/$1 ;;
  esac
}

cnpy-nuwapkg(){ echo $DYB/NuWa-trunk/dybgaudi/Utilities/cnpy ; }
cnpy-nuwapkg-cd(){  cd $(cnpy-nuwapkg); }
cnpy-nuwapkg-action(){
   local action=${1:-ls}
   local iwd=$PWD
   cnpy-scd
   local src
   local dest

   local pkg=$(cnpy-nuwapkg)
   cnpy-names | while read src ; do 
       dest=$pkg/$(cnpy-mapping $src)
       [ ! -d "$(dirname $dest)" ] && mkdir -p $(dirname $dest) 
       case $action in 
         cpto) cp $src $dest ;;
         cpfr) cp $dest $src ;;
            *) echo $action $src $dest && $action $src $dest ;;
       esac 
   done   
   cd $iwd
}
cnpy-nuwapkg-cpto(){ cnpy-nuwapkg-action cpto  ;}
cnpy-nuwapkg-cpfr(){ cnpy-nuwapkg-action cpfr  ;}
cnpy-nuwapkg-diff(){ cnpy-nuwapkg-action diff  ;}
cnpy-nuwapkg-ls(){   cnpy-nuwapkg-action ls  ;}



