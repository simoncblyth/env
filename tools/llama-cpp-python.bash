# === func-gen- : tools/llama-cpp-python fgp tools/llama-cpp-python.bash fgn llama-cpp-python fgh tools src base/func.bash
llama-cpp-python-source(){   echo ${BASH_SOURCE} ; }
llama-cpp-python-edir(){ echo $(dirname $(llama-cpp-python-source)) ; }
llama-cpp-python-ecd(){  cd $(llama-cpp-python-edir); }
llama-cpp-python-dir(){  echo $LOCAL_BASE/env/tools/llama-cpp-python ; }
llama-cpp-python-cd(){   cd $(llama-cpp-python-dir); }
llama-cpp-python-vi(){   vi $(llama-cpp-python-source) ; }
llama-cpp-python-env(){  elocal- ; }
llama-cpp-python-usage(){ cat << EOU


https://llama-cpp-python.readthedocs.io/en/latest/

https://github.com/abetlen/llama-cpp-python




EOU
}
llama-cpp-python-get(){
   local dir=$(dirname $(llama-cpp-python-dir)) &&  mkdir -p $dir && cd $dir

}
