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



Khronos 3D portability exploratory group
-------------------------------------------

* https://www.khronos.org/3dportability/

Khronos Group Considering Portability API On Top Of Vulkan, Metal, 

* http://www.tomshardware.com/news/khronos-meta-api-vulkan-metal-directx12,33962.html

* https://www.khronos.org/blog/good-things-are-coming-to-vr-mobile-graphics-from-the-khronos-group

.. a new 3D portability exploratory group to discuss and formulate a solution
to enable 3D applications that are portable across Vulkan, DX12 and Metal-based
platforms.


* :google:`Khronos 3D portability exploratory group`


Apple WebGPU
--------------

* https://webkit.org/blog/7380/next-generation-3d-graphics-on-the-web/
* https://webkit.org/blog/7504/webgpu-prototype-and-demos/
* https://webkit.org/demos/webgpu/


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
