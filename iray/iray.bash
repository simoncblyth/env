# === func-gen- : iray/iray fgp iray/iray.bash fgn iray fgh iray
iray-src(){      echo iray/iray.bash ; }
iray-source(){   echo ${BASH_SOURCE:-$(env-home)/$(iray-src)} ; }
iray-vi(){       vi $(iray-source) ; }
iray-env(){      elocal- ; }
iray-usage(){ cat << EOU

NVIDIA iRAY 
============


* http://www.nvidia-arc.com/products/iray/iray-integration-framework.html





EOU
}
iray-dir(){ echo $(local-base)/env/iray/iray-iray ; }
iray-cd(){  cd $(iray-dir); }
iray-mate(){ mate $(iray-dir) ; }
iray-get(){
   local dir=$(dirname $(iray-dir)) &&  mkdir -p $dir && cd $dir

}
