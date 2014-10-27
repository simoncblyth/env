# === func-gen- : numpy/cnpy fgp numpy/cnpy.bash fgn cnpy fgh numpy
cnpy-src(){      echo numpy/cnpy.bash ; }
cnpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cnpy-src)} ; }
cnpy-vi(){       vi $(cnpy-source) ; }
cnpy-env(){      elocal- ; }
cnpy-usage(){ cat << EOU

CNPY
=====

* https://github.com/rogersce/cnpy.git

* Added RPATH setup to CMakeLists, see cnpytest- which 
  takes advantage of this

::

    -- Installing: /usr/local/env/cnpy/lib/libcnpy.dylib
    -- Installing: /usr/local/env/cnpy/lib/libcnpy.a
    -- Installing: /usr/local/env/cnpy/include/cnpy.h
    -- Installing: /usr/local/env/cnpy/bin/mat2npz
    -- Installing: /usr/local/env/cnpy/bin/npy2mat
    -- Installing: /usr/local/env/cnpy/bin/npz2mat



EOU
}
cnpy-prefix(){ echo $(local-base)/env/cnpy ; } 
cnpy-sdir(){ echo $(local-base)/env/numpy/cnpy ; }
cnpy-scd(){  cd $(cnpy-sdir); }

cnpy-get(){
   local dir=$(dirname $(cnpy-sdir)) &&  mkdir -p $dir && cd $dir
   [ ! -d cnpy ] && git clone https://github.com/rogersce/cnpy.git
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


