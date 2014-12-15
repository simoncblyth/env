# === func-gen- : numpy/cnpy fgp numpy/cnpy.bash fgn cnpy fgh numpy
cnpy-src(){      echo numpy/cnpy.bash ; }
cnpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cnpy-src)} ; }
cnpy-vi(){       vi $(cnpy-source) ; }
cnpy-env(){      elocal- ; }
cnpy-usage(){ cat << EOU

CNPY
=====

* https://github.com/rogersce/cnpy.git


G5 : kludge containing folder
------------------------------

::

    blyth@ntugrid5 cnpy]$ ll /home/blyth/local/env/cnpy/include/
    total 20
    -rw-r--r-- 1 blyth blyth 9906 Dec  9 15:52 cnpy.h
    drwxr-xr-x 5 blyth blyth 4096 Dec 15 19:39 ..
    drwxr-xr-x 2 blyth blyth 4096 Dec 15 19:39 .
    [blyth@ntugrid5 cnpy]$ mkdir /home/blyth/local/env/cnpy/include/cnpy
    [blyth@ntugrid5 cnpy]$ cd /home/blyth/local/env/cnpy/include/cnpy
    [blyth@ntugrid5 cnpy]$ ln -s ../cnpy.h 





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


Alternatives
--------------

C
~~

* libnpy-  
* npyreader- 

C++
~~~~

* https://gist.github.com/rezoo/5656056


Casting numpy structured arrays into arrays of structs ?
----------------------------------------------------------

For this to work would need to be in control of the 
struct packing/padding/alignment ans match it to the
numpy dtype.  Maybe numpy source would be illuminating.

* http://www.catb.org/esr/structure-packing/ 

Numpy default is packed (unaligned)
------------------------------------

* http://cython.readthedocs.org/en/latest/src/tutorial/numpy.html

    Also note that NumPy record arrays are by default unaligned, meaning data is
    packed as tightly as possible without considering the alignment preferences of
    the CPU. Such unaligned record arrays corresponds to a Cython packed struct. If
    one uses an aligned dtype, by passing align=True to the dtype constructor, one
    must drop the packed keyword on the struct definition.




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
#cnpy-vers(){ echo simoncblyth ; }

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
cnpy-nuwapkg-cd(){  cd $(cnpy-nuwapkg)/$1 ; }
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
cnpy-nuwapkg-echo(){ cnpy-nuwapkg-action echo  ;}
cnpy-nuwapkg-ls(){   cnpy-nuwapkg-action ls  ;}


cnpy-nuwapkg-env(){
    echo -n
}

cnpy-nuwapkg-make() 
{ 
    local iwd=$PWD;
    cnpy-nuwapkg-env
    cnpy-nuwapkg-cd cmt

    cmt br cmt config

    cmt config
    cmt make
    cd $iwd
}




