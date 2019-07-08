# === func-gen- : ai/keras fgp ai/keras.bash fgn keras fgh ai src base/func.bash
keras-source(){   echo ${BASH_SOURCE} ; }
keras-edir(){ echo $(dirname $(keras-source)) ; }
keras-ecd(){  cd $(keras-edir); }
keras-dir(){  echo $LOCAL_BASE/env/ai/keras ; }
keras-cd(){   cd $(keras-dir); }
keras-vi(){   vi $(keras-source) ; }
keras-env(){  elocal- ; }
keras-usage(){ cat << EOU





EOU
}
keras-get(){
   local dir=$(dirname $(keras-dir)) &&  mkdir -p $dir && cd $dir

}
