# === func-gen- : graphics/nvidia/mdl fgp graphics/nvidia/mdl.bash fgn mdl fgh graphics/nvidia src base/func.bash
mdl-source(){   echo ${BASH_SOURCE} ; }
mdl-edir(){ echo $(dirname $(mdl-source)) ; }
mdl-ecd(){  cd $(mdl-edir); }
mdl-dir(){  echo $LOCAL_BASE/env/graphics/nvidia/mdl ; }
mdl-cd(){   cd $(mdl-dir); }
mdl-vi(){   vi $(mdl-source) ; }
mdl-env(){  elocal- ; }
mdl-usage(){ cat << EOU


NVIDA MDL : Material Definition Language
===========================================

Open source since summer 2018

* https://developer.nvidia.com/mdl-sdk
* https://github.com/NVIDIA/MDL-SDK



EOU
}
mdl-get(){
   local dir=$(dirname $(mdl-dir)) &&  mkdir -p $dir && cd $dir

}
