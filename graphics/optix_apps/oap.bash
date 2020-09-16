# === func-gen- : graphics/optix_apps/oap fgp graphics/optix_apps/oap.bash fgn oap fgh graphics/optix_apps src base/func.bash
oap-source(){   echo ${BASH_SOURCE} ; }
oap-edir(){ echo $(dirname $(oap-source)) ; }
oap-ecd(){  cd $(oap-edir); }
oap-dir(){  echo $LOCAL_BASE/env/graphics/oap/OptiX_Apps ; }
oap-cd(){   cd $(oap-dir); }
oap-vi(){   vi $(oap-source) ; }
oap-env(){  elocal- ; }
oap-usage(){ cat << EOU





EOU
}
oap-get(){
   local dir=$(dirname $(oap-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/NVIDIA/OptiX_Apps 


}
