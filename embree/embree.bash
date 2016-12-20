# === func-gen- : embree/embree fgp embree/embree.bash fgn embree fgh embree
embree-src(){      echo embree/embree.bash ; }
embree-source(){   echo ${BASH_SOURCE:-$(env-home)/$(embree-src)} ; }
embree-vi(){       vi $(embree-source) ; }
embree-env(){      elocal- ; }
embree-usage(){ cat << EOU


Embree : High Performance Ray Tracing Kernels
================================================

* https://embree.github.io
* https://embree.github.io/data/embree-siggraph-2016-final.pdf

The kernels are optimized for photo-realistic rendering on the latest Intel® processors 
with support for SSE, AVX, AVX2, and AVX512



EOU
}
embree-dir(){ echo $(local-base)/env/embree/embree-embree ; }
embree-cd(){  cd $(embree-dir); }
embree-mate(){ mate $(embree-dir) ; }
embree-get(){
   local dir=$(dirname $(embree-dir)) &&  mkdir -p $dir && cd $dir

}
