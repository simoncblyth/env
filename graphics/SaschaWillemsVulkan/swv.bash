# === func-gen- : graphics/SaschaWillemsVulkan/swv fgp graphics/SaschaWillemsVulkan/swv.bash fgn swv fgh graphics/SaschaWillemsVulkan
swv-src(){      echo graphics/SaschaWillemsVulkan/swv.bash ; }
swv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(swv-src)} ; }
swv-vi(){       vi $(swv-source) ; }
swv-env(){      elocal- ; }
swv-usage(){ cat << EOU



* https://github.com/SaschaWillems/Vulkan

* see also vgp-

A Great Resource : for Learning Vulkan
---------------------------------------------

Lots about Vulkan, many examples

https://www.saschawillems.de/?p=2770
https://www.saschawillems.de/?page_id=2017
https://www.saschawillems.de/wp-content/uploads/2018/01/Khronos_meetup_munich_fromGLtoVulkan.pdf




EOU
}
swv-dir(){ echo $(local-base)/env/graphics/SaschaWillems/Vulkan; }
swv-cd(){  cd $(swv-dir); }
swv-get(){
   local dir=$(dirname $(swv-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d Vulkan ] && git clone --recursive https://github.com/SaschaWillems/Vulkan.git
}



