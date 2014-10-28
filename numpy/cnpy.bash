# === func-gen- : numpy/cnpy fgp numpy/cnpy.bash fgn cnpy fgh numpy
cnpy-src(){      echo numpy/cnpy.bash ; }
cnpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cnpy-src)} ; }
cnpy-vi(){       vi $(cnpy-source) ; }
cnpy-env(){      elocal- ; }
cnpy-usage(){ cat << EOU

CNPY
=====

* https://github.com/rogersce/cnpy.git

* TODO: evaluate the changes in fork https://github.com/dstahlke/cnpy


D cmake install
------------------

* Added RPATH setup to CMakeLists, see cnpytest- which 
  takes advantage of this

::

    -- Installing: /usr/local/env/cnpy/lib/libcnpy.dylib
    -- Installing: /usr/local/env/cnpy/lib/libcnpy.a
    -- Installing: /usr/local/env/cnpy/include/cnpy.h
    -- Installing: /usr/local/env/cnpy/bin/mat2npz
    -- Installing: /usr/local/env/cnpy/bin/npy2mat
    -- Installing: /usr/local/env/cnpy/bin/npz2mat


How to add to NuWa ?
----------------------

Could be an external, but I suspect I will 
want to make extensive changes if using more
extensively.  So maybe add to Utilities

::

    [blyth@belle7 dybgaudi]$ cnpy-get
    Initialized empty Git repository in /data1/env/local/env/numpy/cnpy/.git/
    Cannot get remote repository information.
    Perhaps git-update-server-info needs to be run there?
    [blyth@belle7 numpy]$ 
    [blyth@belle7 numpy]$ 
    [blyth@belle7 numpy]$ which git 
    /usr/bin/git




EOU
}
cnpy-prefix(){ echo $(local-base)/env/cnpy ; } 
cnpy-sdir(){ echo $(local-base)/env/numpy/cnpy ; }
cnpy-scd(){  cd $(cnpy-sdir); }

cnpy-get(){
   local dir=$(dirname $(cnpy-sdir)) &&  mkdir -p $dir && cd $dir
   [ ! -d cnpy ] &&  git clone git://github.com/simoncblyth/cnpy.git
   # git clone https://github.com/rogersce/cnpy.git  doesnt clone on N 
}


cnpy-build(){

   cnpy-scd

   mkdir build
   cd build

   cmake -DCMAKE_INSTALL_PREFIX=$(cnpy-prefix) ..

   make 
   make install

   cd ..
   rm -rf build


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



