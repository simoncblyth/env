# === func-gen- : graphics/bgfx/bgfx fgp graphics/bgfx/bgfx.bash fgn bgfx fgh graphics/bgfx src base/func.bash
bgfx-source(){   echo ${BASH_SOURCE} ; }
bgfx-edir(){ echo $(dirname $(bgfx-source)) ; }
bgfx-ecd(){  cd $(bgfx-edir); }
bgfx-dir(){  echo $LOCAL_BASE/env/graphics/bgfx/bgfx ; }
bgfx-cd(){   cd $(bgfx-dir); }
bgfx-vi(){   vi $(bgfx-source) ; }
bgfx-env(){  elocal- ; }
bgfx-usage(){ cat << EOU

bgfx
=======

Cross-platform, graphics API agnostic, "Bring Your Own Engine/Framework" style
rendering library.

* https://github.com/bkaradzic/bgfx
* https://bkaradzic.github.io/bgfx/overview.html
* https://twitter.com/search?q=%23madewithbgfx&f=live

  includes Project Aero (augmented reality authoring tool from Adobe) 


Observations
--------------

* Lots of backends 
* Shader language subset ?


See Also
---------

* llgl-
* dileng-


Introductions
---------------


* https://github.com/jpcy/bgfx-minimal-example

  ..doesn't use the bgfx example framework. GLFW is used for windowing. There
  are separate single and multithreaded examples. 

  Premake 5 is used instead of GENie, so this also serves as an example of how
  to build bgfx, bimg and bx with a different build system.

* https://dev.to/pperon/hello-bgfx-4dka

  Rotating cube, Direct3D backend

* https://www.sandeepnambiar.com/getting-started-with-bgfx/

  Using the bgfx library with C++ on Ubuntu

  Linux, SDL based



Genie
------


Premake
----------

* https://premake.github.io

In addition to its project generation capabilities, Premake also provides a
complete Lua scripting environment, enabling the automation of complex
configuration tasks such as setting up new source tree checkouts or creating
deployment packages. These scripts will run on any platform, ending batch/shell
script duplication.

Premake is a "plain old C" application, distributed as a single executable
file. It is small, weighing in at around 200K. It does not require any
additional libraries or runtimes to be installed, and should build and run
pretty much anywhere. It is currently being tested and used on Windows, Mac OS
X, Linux, and other POSIX environments. It uses only a handful of platform
dependent routines (directory management, mostly). Adding support for
additional toolsets and languages is straightforward. The source code is
available under the BSD License. The source code is hosted right here on
GitHub; file downloads are currently hosted on SourceForge.



RendererType
--------------

bgfx.h::

  49     struct RendererType
  50     {
  51         /// Renderer types:
  52         enum Enum
  53         {
  54             Noop,         //!< No rendering.
  55             Direct3D9,    //!< Direct3D 9.0
  56             Direct3D11,   //!< Direct3D 11.0
  57             Direct3D12,   //!< Direct3D 12.0
  58             Gnm,          //!< GNM
  59             Metal,        //!< Metal
  60             Nvn,          //!< NVN
  61             OpenGLES,     //!< OpenGL ES 2.0+
  62             OpenGL,       //!< OpenGL 2.1+
  63             Vulkan,       //!< Vulkan
  64             WebGPU,       //!< WebGPU
  65 
  66             Count
  67         };
  68     };



WebGPU
    https://gpuweb.github.io/gpuweb/

    WebGPU is an API that exposes the capabilities of GPU hardware for the Web. The
    API is designed from the ground up to efficiently map to the Vulkan, Direct3D
    12, and Metal native GPU APIs. WebGPU is not related to WebGL and does not
    explicitly target OpenGL ES.


PlatformData
-------------

bgfx.h::

     609     struct PlatformData
     610     {
     611         PlatformData();
     612 
     613         void* ndt;          //!< Native display type (*nix specific).
     614         void* nwh;          //!< Native window handle. If `NULL` bgfx will create headless
     615                             ///  context/device if renderer API supports it.
     616         void* context;      //!< GL context, or D3D device. If `NULL`, bgfx will create context/device.
     617         void* backBuffer;   //!< GL back-buffer, or D3D render target view. If `NULL` bgfx will
     618                             ///  create back-buffer color surface.
     619         void* backBufferDS; //!< Backbuffer depth/stencil. If `NULL` bgfx will create back-buffer
     620                             ///  depth/stencil surface.
     621     };


CrossPlatform Shader Language
-------------------------------


* https://bkaradzic.github.io/bgfx/tools.html#shader-compiler-shaderc

Shader Compiler (shaderc)

bgfx cross-platform shader language is based on GLSL syntax. It’s uses ANSI C
preprocessor to transform GLSL like language syntax into HLSL. This technique
has certain drawbacks, but overall it’s simple and allows quick authoring of
cross-platform shaders.

* https://github.com/bkaradzic/bgfx/blob/master/src/bgfx_shader.sh

Issue : Geometry shaders not supported
-----------------------------------------

* https://github.com/bkaradzic/bgfx/issues/1543
* https://github.com/bkaradzic/bgfx/issues/332

bkaradzic commented on Apr 11, 2015

If you're working on very specific tool you could add support easily because
you can focus only on specific renderer.

In order for me to add feature I want to make sure it works everywhere or at
least on most platforms in cross-platform way.

Also a lot of those things you mentioned you can achieve with existing system
(some combination of compute + vertex shaders).

See also:
http://www.joshbarczak.com/blog/?p=667

* *replace photon propagation geometry shaders with compute + vertex for compatibility*
* NEED TO GET SOME EXPERIENCE WITH USING COMPUTE+VERTEX SHADERS TOGETHER



EOU
}
bgfx-get(){
   local dir=$(dirname $(bgfx-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d bx ]   &&  git clone git://github.com/bkaradzic/bx.git
   [ ! -d bimg ] &&  git clone git://github.com/bkaradzic/bimg.git
   [ ! -d bgfx ] &&  git clone git://github.com/bkaradzic/bgfx.git 
}

bgfx-make(){
   bgfx-cd
   make osx-debug64  
}



