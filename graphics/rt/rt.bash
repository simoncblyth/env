# === func-gen- : graphics/rt/rt fgp graphics/rt/rt.bash fgn rt fgh graphics/rt src base/func.bash
rt-source(){   echo ${BASH_SOURCE} ; }
rt-edir(){ echo $(dirname $(rt-source)) ; }
rt-ecd(){  cd $(rt-edir); }
rt-dir(){  echo $LOCAL_BASE/env/graphics/rt/rt ; }
rt-cd(){   cd $(rt-dir); }
rt-vi(){   vi $(rt-source) ; }
rt-env(){  elocal- ; }
rt-usage(){ cat << EOU

RT : Ray Tracing APIs
=======================

* http://www.cgchannel.com/2020/03/vulkan-now-supports-ray-tracing/

* https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing/


* https://news.developer.nvidia.com/ray-tracing-essentials-part-4-the-ray-tracing-pipeline/


* :google:`ray tracing procedural primitives`

* https://arxiv.org/abs/2012.10357
* ~/opticks_refs/proceduray_2012_10357.pdf


* :google:`ray tracing vulkan dxr optix embree metal`


* https://www.escape-technology.com/news/total-chaos-and-real-time-ray-tracing

* https://www.willusher.io/blog


Books
------


* https://www.realtimerendering.com/raytracing.html#books



RTX
----

* https://developer.nvidia.com/blog/rtx-best-practices/
* https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing/


NVIDIA OptiX
-------------- 

Microsoft DXR (Direct X Ray Tracing)
---------------------------------------

Vulkan Ray Tracing Extensions
---------------------------------

* https://www.khronos.org/blog/ray-tracing-in-vulkan
* https://www.khronos.org/blog/vulkan-ray-tracing-best-practices-for-hybrid-rendering


* https://forums.developer.nvidia.com/t/vk-nv-raytracing-with-procedural-geometries/71065
* VK_GEOMETRY_TYPE_AABBS_NV

* https://nvpro-samples.github.io/vk_raytracing_tutorial/vkrt_tuto_intersection.md.html


* https://www.gamasutra.com/blogs/EgorYusov/20210106/375829/A_Diligent_Approach_to_Ray_Tracing.php

* https://github.com/GPSnoopy/RayTracingInVulkan


Metal Ray Tracing
---------------------

Embree
---------





EOU
}
rt-get(){
   local dir=$(dirname $(rt-dir)) &&  mkdir -p $dir && cd $dir

}
