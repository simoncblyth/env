# === func-gen- : graphics/vulkan/vulkanrt fgp graphics/vulkan/vulkanrt.bash fgn vulkanrt fgh graphics/vulkan src base/func.bash
vulkanrt-source(){   echo ${BASH_SOURCE} ; }
vulkanrt-edir(){ echo $(dirname $(vulkanrt-source)) ; }
vulkanrt-ecd(){  cd $(vulkanrt-edir); }
vulkanrt-dir(){  echo $LOCAL_BASE/env/graphics/vulkan/vulkanrt ; }
vulkanrt-cd(){   cd $(vulkanrt-dir); }
vulkanrt-vi(){   vi $(vulkanrt-source) ; }
vulkanrt-env(){  elocal- ; }
vulkanrt-usage(){ cat << EOU


Vulkan RT Extensions
=========================


* https://www.khronos.org/blog/ray-tracing-in-vulkan
* https://github.com/GPSnoopy/RayTracingInVulkan


Quake II engine with real-time path tracing.
----------------------------------------------

* http://brechpunkt.de/q2vkpt/
* https://github.com/cschied/q2vkpt/
   


EOU
}
vulkanrt-get(){
   local dir=$(dirname $(vulkanrt-dir)) &&  mkdir -p $dir && cd $dir

}
