# === func-gen- : cuda/cuda_samples fgp cuda/cuda_samples.bash fgn cuda_samples fgh cuda
cuda_samples-src(){      echo cuda/cuda_samples.bash ; }
cuda_samples-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cuda_samples-src)} ; }
cuda_samples-vi(){       vi $(cuda_samples-source) ; }
cuda_samples-env(){      elocal- ; }
cuda_samples-usage(){ cat << EOU





EOU
}
cuda_samples-dir(){ echo $(local-base)/env/cuda/cuda-cuda_samples ; }
cuda_samples-cd(){  cd $(cuda_samples-dir); }
cuda_samples-mate(){ mate $(cuda_samples-dir) ; }
cuda_samples-get(){
   local dir=$(dirname $(cuda_samples-dir)) &&  mkdir -p $dir && cd $dir

}
