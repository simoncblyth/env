# === func-gen- : graphics/llgl/llgl fgp graphics/llgl/llgl.bash fgn llgl fgh graphics/llgl src base/func.bash
llgl-source(){   echo ${BASH_SOURCE} ; }
llgl-edir(){ echo $(dirname $(llgl-source)) ; }
llgl-ecd(){  cd $(llgl-edir); }
llgl-dir(){  echo $LOCAL_BASE/env/graphics/llgl/LLGL ; }
llgl-cd(){   cd $(llgl-dir); }
llgl-vi(){   vi $(llgl-source) ; }
llgl-env(){  elocal- ; }
llgl-usage(){ cat << EOU


LLGL : Low Level Graphics Library
====================================

* https://github.com/LukasBanana/LLGL

Low Level Graphics Library (LLGL) is a thin abstraction layer for the modern
graphics APIs OpenGL, Direct3D, Vulkan, and Metal


See Also
-----------

* bgfx-
* dileng-


Others
---------

* https://lucidindex.com/category/cpp/graphics

* :google:`llgl bgfx diligent engine`

* https://www.gamasutra.com/blogs/EgorYusov/20171130/310274/Designing_a_Modern_CrossPlatform_LowLevel_Graphics_Library.php

* https://www.khronos.org/news/categories/category/vulkan

* https://awesomeopensource.com/projects/vulkan


EOU
}
llgl-get(){
   local dir=$(dirname $(llgl-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d LLGL ] && git clone https://github.com/LukasBanana/LLGL


}
