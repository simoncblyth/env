# === func-gen- : graphics/nvidia/vxgi fgp graphics/nvidia/vxgi.bash fgn vxgi fgh graphics/nvidia
vxgi-src(){      echo graphics/nvidia/vxgi.bash ; }
vxgi-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vxgi-src)} ; }
vxgi-vi(){       vi $(vxgi-source) ; }
vxgi-env(){      elocal- ; }
vxgi-usage(){ cat << EOU


* https://developer.nvidia.com/vxgi

* http://www.geforce.com/whats-new/articles/maxwells-voxel-global-illumination-technology-introduces-gamers-to-the-next-generation-of-graphics
* https://research.nvidia.com/sites/default/files/publications/GIVoxels-pg2011-authors.pdf
* http://www.geforce.com/whats-new/articles/maxwells-voxel-global-illumination-technology-introduces-gamers-to-the-next-generation-of-graphics





EOU
}
vxgi-dir(){ echo $(local-base)/env/graphics/nvidia/graphics/nvidia-vxgi ; }
vxgi-cd(){  cd $(vxgi-dir); }
vxgi-mate(){ mate $(vxgi-dir) ; }
vxgi-get(){
   local dir=$(dirname $(vxgi-dir)) &&  mkdir -p $dir && cd $dir

}
