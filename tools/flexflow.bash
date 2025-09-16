# === func-gen- : tools/flexflow fgp tools/flexflow.bash fgn flexflow fgh tools src base/func.bash
flexflow-source(){   echo ${BASH_SOURCE} ; }
flexflow-edir(){ echo $(dirname $(flexflow-source)) ; }
flexflow-ecd(){  cd $(flexflow-edir); }
flexflow-dir(){  echo $LOCAL_BASE/env/tools/flexflow ; }
flexflow-cd(){   cd $(flexflow-dir); }
flexflow-vi(){   vi $(flexflow-source) ; }
flexflow-env(){  elocal- ; }
flexflow-usage(){ cat << EOU

https://github.com/flexflow/flexflow-serve

https://github.com/flexflow/flexflow-serve/blob/inference/SERVE.md



EOU
}
flexflow-get(){
   local dir=$(dirname $(flexflow-dir)) &&  mkdir -p $dir && cd $dir

}
