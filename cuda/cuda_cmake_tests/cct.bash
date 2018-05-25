# === func-gen- : cuda/cuda_cmake_tests/cct fgp cuda/cuda_cmake_tests/cct.bash fgn cct fgh cuda/cuda_cmake_tests
cct-src(){      echo cuda/cuda_cmake_tests/cct.bash ; }
cct-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cct-src)} ; }
cct-vi(){       vi $(cct-source) ; }
cct-env(){      elocal- ; }
cct-usage(){ cat << EOU





EOU
}
cct-dir(){ echo $(local-base)/env/cuda/cuda_cmake_tests/cuda/cuda_cmake_tests-cct ; }
cct-cd(){  cd $(cct-dir); }
cct-mate(){ mate $(cct-dir) ; }
cct-get(){
   local dir=$(dirname $(cct-dir)) &&  mkdir -p $dir && cd $dir

}
