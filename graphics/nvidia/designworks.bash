# === func-gen- : graphics/nvidia/designworks fgp graphics/nvidia/designworks.bash fgn designworks fgh graphics/nvidia
designworks-src(){      echo graphics/nvidia/designworks.bash ; }
designworks-source(){   echo ${BASH_SOURCE:-$(env-home)/$(designworks-src)} ; }
designworks-vi(){       vi $(designworks-source) ; }
designworks-env(){      elocal- ; }
designworks-usage(){ cat << EOU

NVIDIA Designworks
====================

* https://developer.nvidia.com/designworks

* Uses common NVIDIA Registered developer login


Solutions for creating realistic images using ray tracing, or real time rendering techniques with OpenGL®, DirectX®, or Vulkan™

*Iray SDK*
       rendering by simulating the physical behavior of light and materials
*MDL SDK*
       description language for materials and lights that can be shared across renderers
*vMaterials*
       library with hundreds of ready to use real world materials
*OptiX*
       programmable ray tracing framework
*VXGI*
       real-time global illumination
*Path Rendering*
       GPU-acceleration for 2D vector graphics
*NVIDIA Pro Pipeline*
       high performance rendering pipeline for scenes with complex scene graphs
Rendering Methods optimized sampling algorithms for faster image convergence, modified shading and more





EOU
}
designworks-dir(){ echo $(local-base)/env/graphics/nvidia/graphics/nvidia-designworks ; }
designworks-cd(){  cd $(designworks-dir); }
designworks-mate(){ mate $(designworks-dir) ; }
designworks-get(){
   local dir=$(dirname $(designworks-dir)) &&  mkdir -p $dir && cd $dir

}
