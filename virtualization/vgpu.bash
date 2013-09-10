# === func-gen- : virtualization/vgpu fgp virtualization/vgpu.bash fgn vgpu fgh virtualization
vgpu-src(){      echo virtualization/vgpu.bash ; }
vgpu-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vgpu-src)} ; }
vgpu-vi(){       vi $(vgpu-source) ; }
vgpu-env(){      elocal- ; }
vgpu-usage(){ cat << EOU

VGPU
=====

* http://www.zillians.com/how-it-works-2/




EOU
}
vgpu-dir(){ echo $(local-base)/env/virtualization/virtualization-vgpu ; }
vgpu-cd(){  cd $(vgpu-dir); }
vgpu-mate(){ mate $(vgpu-dir) ; }
vgpu-get(){
   local dir=$(dirname $(vgpu-dir)) &&  mkdir -p $dir && cd $dir

}
