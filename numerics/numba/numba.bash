# === func-gen- : numerics/numba/numba fgp numerics/numba/numba.bash fgn numba fgh numerics/numba
numba-src(){      echo numerics/numba/numba.bash ; }
numba-source(){   echo ${BASH_SOURCE:-$(env-home)/$(numba-src)} ; }
numba-vi(){       vi $(numba-source) ; }
numba-env(){      elocal- ; }
numba-usage(){ cat << EOU

Numba : python cuda acceleration
==================================

* https://devblogs.nvidia.com/numba-python-cuda-acceleration/
* https://devblogs.nvidia.com/seven-things-numba/







EOU
}
numba-dir(){ echo $(local-base)/env/numerics/numba/numerics/numba-numba ; }
numba-cd(){  cd $(numba-dir); }
numba-mate(){ mate $(numba-dir) ; }
numba-get(){
   local dir=$(dirname $(numba-dir)) &&  mkdir -p $dir && cd $dir

}
