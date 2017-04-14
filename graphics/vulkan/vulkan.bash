# === func-gen- : graphics/vulkan/vulkan fgp graphics/vulkan/vulkan.bash fgn vulkan fgh graphics/vulkan
vulkan-src(){      echo graphics/vulkan/vulkan.bash ; }
vulkan-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vulkan-src)} ; }
vulkan-vi(){       vi $(vulkan-source) ; }
vulkan-env(){      elocal- ; }
vulkan-usage(){ cat << EOU

Vulkan
=======

C++ Bindings
-------------

* https://developer.nvidia.com/vulkan-c-bindings-reloaded
* https://github.com/KhronosGroup/Vulkan-Hpp

VR
---

* https://developer.nvidia.com/getting-vulkan-ready-vr

Driver
----------

* https://developer.nvidia.com/nvidia-gdc-vulkan-driver-available-now


EOU
}
vulkan-dir(){ echo $(local-base)/env/graphics/vulkan ; }
vulkan-cd(){  cd $(vulkan-dir); }
vulkan-get(){
   local dir=$(dirname $(vulkan-dir)) &&  mkdir -p $dir && cd $dir

}
