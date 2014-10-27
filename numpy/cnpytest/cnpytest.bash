# === func-gen- : numpy/cnpytest/cnpytest fgp numpy/cnpytest/cnpytest.bash fgn cnpytest fgh numpy/cnpytest
cnpytest-src(){      echo numpy/cnpytest/cnpytest.bash ; }
cnpytest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cnpytest-src)} ; }
cnpytest-vi(){       vi $(cnpytest-source) ; }
cnpytest-env(){      
   elocal- 
   geant4sys- 
}
cnpytest-usage(){ cat << EOU

CNPYTEST
==========

Example usage of CNPY, writing NPY array 
serializations from C++

Read from ipython::

   import numpy as np
   a = np.load("arr1.npy")



EOU
}
cnpytest-dir(){ echo $(env-home)/numpy/cnpytest ; }
cnpytest-cd(){  cd $(cnpytest-dir); }
cnpytest-prefix(){ echo $(local-base)/env ; }

cnpytest-build(){

   cnpytest-cd

   mkdir build
   cd build

   cmake .. -DCMAKE_INSTALL_PREFIX=$(cnpytest-prefix) -DGeant4_DIR=$(geant4sys-cmakedir) 

   make 
   make install

   cd ..
   rm -rf build

}





