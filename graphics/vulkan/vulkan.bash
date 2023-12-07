# === func-gen- : graphics/vulkan/vulkan fgp graphics/vulkan/vulkan.bash fgn vulkan fgh graphics/vulkan
vulkan-src(){      echo graphics/vulkan/vulkan.bash ; }
vulkan-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vulkan-src)} ; }
vulkan-vi(){       vi $(vulkan-source) ; }
vulkan-env(){      elocal- ; }
vulkan-usage(){ cat << EOU

Vulkan
=======



Lumibench
-----------

* https://people.ece.ubc.ca/~aamodt/publications/papers/lumibench.iiswc2023.pdf
* ~/opticks_refs/lumibench.iiswc2023.pdf


Ray Tracing Extension
-----------------------

* https://www.khronos.org/blog/ray-tracing-in-vulkan



Provisional Ray Tracing Extension 
-------------------------------------

* https://www.khronos.org/news/press/khronos-group-releases-vulkan-ray-tracing
* https://geant4.web.cern.ch/node/1886


Ray Tracing in Vulkan with NVIDIA RTX Extension
--------------------------------------------------

* https://github.com/GPSnoopy/RayTracingInVulkan


Resource Lists
----------------

* https://www.geeks3d.com/20160205/vulkan-programming-resources-list/



Luminaries
-----------

* https://www.saschawillems.de/


Official Samples
-------------------

* https://www.khronos.org/blog/vulkan-releases-unified-samples-repository
* https://github.com/khronosGroup/Vulkan-samples

NVIDIA
-------

* https://developer.nvidia.com/opengl-vulkan


C++ Bindings
-------------

* https://developer.nvidia.com/vulkan-c-bindings-reloaded
* https://github.com/KhronosGroup/Vulkan-Hpp


Vulkan Tutorial
----------------

* https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers
* https://vulkan-tutorial.com/Development_environment


MoltenVK with GLFW
--------------------

* https://github.com/glfw/glfw/issues/870


LunarG Vulkan SDK
-------------------

* https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html
* https://vulkan.lunarg.com/sdk/home#mac
* https://github.com/LunarG


MoltenVK goes open source, Feb 2018
-------------------------------------

* https://www.neowin.net/news/moltenvk-popular-vulkan-development-tool-for-macos-goes-open-source
* https://www.anandtech.com/show/12465/khronos-group-extends-vulkan-portability-with-opensource
* https://arstechnica.com/gadgets/2018/02/vulkan-is-coming-to-macos-ios-but-no-thanks-to-apple/


Vulkan Portability Initiative
-------------------------------

Tools and Libraries for Bringing Vulkan Applications to macOS and iOS

* https://www.khronos.org/vulkan/portability-initiative


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
