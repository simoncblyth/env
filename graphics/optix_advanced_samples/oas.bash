# === func-gen- : graphics/optix_advanced_samples/oas fgp graphics/optix_advanced_samples/oas.bash fgn oas fgh graphics/optix_advanced_samples
oas-src(){      echo graphics/optix_advanced_samples/oas.bash ; }
oas-source(){   echo ${BASH_SOURCE:-$(env-home)/$(oas-src)} ; }
oas-vi(){       vi $(oas-source) ; }
oas-env(){      elocal- ; }
oas-usage(){ cat << EOU


OptiX Advanced Samples
========================

* https://devtalk.nvidia.com/default/topic/998546/optix/optix-advanced-samples-on-github/

* http://on-demand.gputechconf.com/gtc/2018/presentation/s8518-an-introduction-to-optix.pdf

* http://on-demand.gputechconf.com/gtc/2018/video/S8518/

  1hr 14min


An Introduction to NVIDIA OptiX
Ankit Patel (NVIDIA), Detlef Roettger (NVIDIA)

0:17:49
    setDevices 
    OptiX 4+ support heterogeneous, will build shaders for each architecture

0:20:00
    stack size as small as possible, eg 1000 bytes
    (as with 5120 cores it becomes large)
    (planning to change this API to set maximum recursions)


OptiX profiling with Nsight Compute
Johann Korndoerfer (NVIDIA)



:google:`OptiX benchmark`
---------------------------


* https://devtalk.nvidia.com/default/topic/1047464/optix/rtx-on-off-benchmark-optix-6/

* https://www.reddit.com/r/nvidia/comments/9jfjen/10_gigarays_translate_to_32_gigarays_in_real/


* http://boostclock.com/show/000219/gpu-rendering-nv-fermat-gtx980ti-gtx1080-gtx1080ti-titanv.html

* http://boostclock.com/show/000230/gpu-rendering-nv-gigarays-gtx980ti-gtx1080-gtx1080ti-rtx2080.html


fermat (OptiX prime only?) not a good benchmark for Turing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


* https://github.com/NVlabs/fermat

* https://nvlabs.github.io/fermat/

* https://github.com/NVlabs/fermat/blob/master/src/rt.cpp

* https://devtalk.nvidia.com/default/topic/1039326/optix/nvidia-fermat-optix-rtx-ray-tracing-benchmark-nvidia-titan-v-gtx-1080-ti-gtx-1080-gtx-980-ti/

  OptiX Prime does not take advantage of RT Cores so Fermat is not a good benchmark for Turing. 


GPU Path tracing
------------------

* https://github.com/straaljager/

* http://raytracey.blogspot.com/2019/02/nvidia-release-optix-60-with-support.html



Github search for OptiX
-------------------------

* https://github.com/search?o=desc&q=OptiX&s=forks&type=Repositories

* https://github.com/knightcrawler25/Optix-PathTracer

* https://github.com/shocker-0x15/VLR

* https://github.com/ozen/PyOptiX

* https://github.com/ozen/optix-docker



Samples
---------

optixParticleVolumes
    nice 3D interaction, galaxy ? 
optixGlass
    glass teapot 
optixHello
optixOcean
optixProgressivePhotonMap
optixVox
    all these are working without CUDA_VISIBLE_DEVICES being set 


optixIntro_01
    CUDA_VISIBLE_DEVICES=0 gdb ./optixIntro_01 

    SEGV at launch using either GPU

    #6  0x00007fffee7b2996 in ?? () from /lib64/libnvoptix.so.1
    #7  0x000000000040e35e in launch (image_height=<optimized out>, image_width=<optimized out>, entry_point_index=0, this=0xa041e0) at /usr/local/OptiX_600/include/optixu/optixpp_namespace.h:2901
    #8  Application::render (this=0x861890) at /home/blyth/local/env/graphics/oas/optix_advanced_samples/src/optixIntroduction/optixIntro_01/src/Application.cpp:417
    #9  0x000000000040979f in main (argc=1, argv=<optimized out>) at /home/blyth/local/env/graphics/oas/optix_advanced_samples/src/optixIntroduction/optixIntro_01/src/main.cpp:225
    (gdb) 


optixIntro_02
    Gradient

optixIntro_03
    Sphere and plane

optixIntro_04
    Box, Sphere and Torus (but its tesselated)

optixIntro_05
    Box, Sphere and Torus again with interactive materials 

optixIntro_06
    Again with more material options



All the optixIntro fail with::

   Invalid device (Details: Function "RTresult _rtContextSetDevices(RTcontext, unsigned int, const int*)" caught exception: All devices must be RTCore capable.)

Unless pick single GPU with:: 

   CUDA_VISIBLE_DEVICES=0 
   CUDA_VISIBLE_DEVICES=1


::

    blyth@localhost bin]$ CUDA_VISIBLE_DEVICES=0,1 ./optixIntro_04
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7.0
      Total Memory: 12621381632
      Clock Rate: 1455000 kHz
      Max. Threads per Block: 1024
      Streaming Multiprocessor Count: 80
      Execution Timeout Enabled: 0
      Max. Hardware Texture Count: 1048576
      TCC Driver enabled: 0
      CUDA Device Ordinal: 0

    Device 1: TITAN RTX
      Compute Support: 7.5
      Total Memory: 25364987904
      Clock Rate: 1770000 kHz
      Max. Threads per Block: 1024
      Streaming Multiprocessor Count: 72
      Execution Timeout Enabled: 1
      Max. Hardware Texture Count: 1048576
      TCC Driver enabled: 0
      CUDA Device Ordinal: 1

    Number of Devices = 2

    Invalid device (Details: Function "RTresult _rtContextSetDevices(RTcontext, unsigned int, const int*)" caught exception: All devices must be RTCore capable.)
    Error: 4: Application initialization failed.
    [blyth@localhost bin]$ 




Externals Warnings
-------------------

OpenGL warning
~~~~~~~~~~~~~~~~~~

::

    CMake Warning (dev) at /usr/share/cmake3/Modules/FindOpenGL.cmake:270 (message):
      Policy CMP0072 is not set: FindOpenGL prefers GLVND by default when
      available.  Run "cmake --help-policy CMP0072" for policy details.  Use the
      cmake_policy command to set the policy and suppress this warning.

      FindOpenGL found both a legacy GL library:

        OPENGL_gl_LIBRARY: /usr/lib64/libGL.so

      and GLVND libraries for OpenGL and GLX:

        OPENGL_opengl_LIBRARY: /usr/lib64/libOpenGL.so
        OPENGL_glx_LIBRARY: /usr/lib64/libGLX.so

      OpenGL_GL_PREFERENCE has not been set to "GLVND" or "LEGACY", so for
      compatibility with CMake 3.10 and below the legacy GL library will be used.
    Call Stack (most recent call first):
      CMakeLists.txt:132 (find_package)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    -- Found OpenGL: /usr/lib64/libOpenGL.so   


DevIL missing
~~~~~~~~~~~~~~~~

* http://openil.sourceforge.net/
* https://github.com/DentonW/DevIL


::

    -- Could NOT find DevIL (missing: IL_LIBRARIES ILU_LIBRARIES IL_INCLUDE_DIR) 
    CMake Deprecation Warning at support/glfw/CMakeLists.txt:10 (cmake_policy):
      The OLD behavior for policy CMP0042 will be removed from a future version
      of CMake.

      The cmake-policies(7) manual explains that the OLD behaviors of all
      policies are deprecated and that a policy should be set to OLD only under
      specific short-term circumstances.  Projects should be ported to the NEW
      behavior and not rely on setting a policy to OLD.

::

     34 add_subdirectory(optixIntro_06)
     35 if (IL_FOUND)
     36   add_subdirectory(optixIntro_07)
     37   add_subdirectory(optixIntro_08)
     38   add_subdirectory(optixIntro_09)
     39   add_subdirectory(optixIntro_10)
     40 else()
     41   message(WARNING "DevIL image library not found. Please set IL_LIBRARIES, ILU_LIBRARIES, ILUT_LIBRARIES, and IL_INCLUDE_DIR to build OptiX introduction samples 07 to 10.")
     42 endif()





EOU
}
oas-dir(){ echo $(local-base)/env/graphics/oas/optix_advanced_samples ; }
oas-cd(){  cd $(oas-dir); }
oas-c(){   cd $(oas-dir); }

oas-get(){
   local dir=$(dirname $(oas-dir)) &&  mkdir -p $dir && cd $dir

   #local url=https://github.com/nvpro-samples/optix_advanced_samples
   local url=git@github.com:simoncblyth/optix_advanced_samples.git

   [ ! -d $(basename $url) ] && git clone $url

}


oas-sdir(){ echo $(oas-dir)/src ; }
oas-bdir(){ echo $(oas-dir).build ; }
oas-xdir(){ echo $(oas-bdir)/bin ; }

oas-scd(){  cd $(oas-sdir) ; }
oas-bcd(){  cd $(oas-bdir) ; }
oas-xcd(){  cd $(oas-xdir) ; }

oas-cmake(){
    local iwd=$PWD
    local bdir=$(oas-bdir) 
    local sdir=$(oas-sdir)

    mkdir -p $bdir && cd $bdir

    optix-

    devil-
    cmake $sdir \
          -DOptiX_INSTALL_DIR=$(optix-install-dir) \
          -DIL_LIBRARIES=$(devil-lib IL) \
          -DILU_LIBRARIES=$(devil-lib ILU) \
          -DILUT_LIBRARIES=$(devil-lib ILUT) \
          -DIL_INCLUDE_DIR=$(devil-include-dir) \

# DevIL image library not found. Please set IL_LIBRARIES, ILU_LIBRARIES, ILUT_LIBRARIES, and IL_INCLUDE_DIR to build OptiX introduction samples 07 to 10
# blyth@localhost optixIntroduction]$ find /usr/share/cmake3/ -name FindDevIL.cmake
#
#   /usr/share/cmake3/Modules/FindDevIL.cmake
#


    cd $iwd
}

oas-make(){
    local iwd=$PWD
    oas-bcd

    make $*

    cd $iwd
}

oas--()
{
    oas-get
    oas-cmake
    oas-make
}


