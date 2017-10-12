# === func-gen- : graphics/egl/egl fgp graphics/egl/egl.bash fgn egl fgh graphics/egl
egl-src(){      echo graphics/egl/egl.bash ; }
egl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(egl-src)} ; }
egl-vi(){       vi $(egl-source) ; }
egl-env(){      elocal- ; }
egl-usage(){ cat << EOU

EGL : OpenGL on headless GPU server nodes
============================================

* https://devblogs.nvidia.com/parallelforall/egl-eye-opengl-visualization-without-x-server/
* https://www.khronos.org/registry/EGL/sdk/docs/man/html/eglIntro.xhtml



Search
--------

* :google:`nvidia egl without x11`

* https://devtalk.nvidia.com/default/topic/947656/create-a-gl-3-context-without-x/?offset=8


demotomohiro : EGL + OpenGL + GLEW (offline rendering)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/1005748/opengl/opengl-without-x-using-egl/

When you use EGL and (desktop)OpenGL with GLEW, you need GLEW 2.0.0 or newer.
And GLEW must be built with::

   make SYSTEM=linux-egl

Here is my working EGL + OpenGL + GLEW initialize/uninitialize code for off-line rendering(Only render to frame buffer object).

* https://github.com/demotomohiro/Reflection-Refraction-less-Ronpa-Raytracing-Renderer/blob/master/src/glcontext_egl.cpp
* https://github.com/demotomohiro/Reflection-Refraction-less-Ronpa-Raytracing-Renderer


Tao EGL ppm demo
~~~~~~~~~~~~~~~~~

Example from Tao.

   I have a try, and it works finally. The attachment is an example.
   DISPLAY should be unset when you ssh, otherwise it would be failed.

::

    $ g++ myfirst.cc -lEGL -lGL && ./a.out
    egl error 12288
    egl error 12288
    major/minor: 1/4
    egl error 12288
    numConfigs: 1

    [simon@localhost egl]$ python -c "print 0x3000 "   ## thats  EGL_SUCCESS
    12288


Mine at SDUGPU (SG)
~~~~~~~~~~~~~~~~~~~~~~~

I get EGL_NOT_INITIALIZED 0x3001::

    [simon@localhost egl]$ egl--
     display 0x8e8030
    eglGetDisplay... err 0x3000
    libEGL warning: DRI2: xcb_connect failed
    libEGL warning: DRI2: xcb_connect failed
    eglInitialize... err 0x3001
    egl error 12288
    major/minor: 0/0
    egl error 12289
    numConfigs: 0
    0 0.000000 
    1 0.005000 
    2 0.010000 

Similar issue fixed by running Xfvb. But the point of is not to need X ?

* https://github.com/klokantech/tileserver-gl/issues/197


Getting all zeros on SG in the ppm::

   [simon@localhost egl]$ xxd -l 100 fig-myfirst00.ppm
   [simon@localhost egl]$ xxd -l 100 fig-myfirst10.ppm
   [simon@localhost egl]$ xxd -l 100 fig-myfirst99.ppm


EGL Wrong Libs ?
~~~~~~~~~~~~~~~~~~~~~

Perhaps are defaulting to the wrong libs ?::

    ldd a.out 

    linux-vdso.so.1 =>  (0x00007fffe3fff000)
    libEGL.so.1 => /usr/lib64/libEGL.so.1 (0x00007f47ad3cb000)
    libGL.so.1 => /usr/lib64/libGL.so.1 (0x0000003832c00000)
    libstdc++.so.6 => /usr/lib64/libstdc++.so.6 (0x000000382ec00000)
    libm.so.6 => /lib64/libm.so.6 (0x000000356b000000)
    libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x000000382f000000)
    libc.so.6 => /lib64/libc.so.6 (0x000000356a400000)
    ...


Huh looks like depending in X11::

    [simon@localhost egl]$ ldd /usr/lib64/libEGL.so.1
        linux-vdso.so.1 =>  (0x00007fffe8552000)
        libselinux.so.1 => /lib64/libselinux.so.1 (0x00007f897a850000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f897a632000)
        libX11-xcb.so.1 => /usr/lib64/libX11-xcb.so.1 (0x00007f897a431000)
        libX11.so.6 => /usr/lib64/libX11.so.6 (0x00007f897a0f4000)
        libxcb-dri2.so.0 => /usr/lib64/libxcb-dri2.so.0 (0x00007f8979eef000)
        libxcb-xfixes.so.0 => /usr/lib64/libxcb-xfixes.so.0 (0x00007f8979ce8000)
        libxcb-render.so.0 => /usr/lib64/libxcb-render.so.0 (0x00007f8979adc000)
        libxcb-shape.so.0 => /usr/lib64/libxcb-shape.so.0 (0x00007f89798d8000)
        libxcb.so.1 => /usr/lib64/libxcb.so.1 (0x00007f89796b3000)
        libgbm.so.1 => /usr/lib64/libgbm.so.1 (0x00007f89794a8000)
        libm.so.6 => /lib64/libm.so.6 (0x00007f8979223000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00007f897901f000)
        libdrm.so.2 => /usr/lib64/libdrm.so.2 (0x00007f8978e12000)
        libexpat.so.1 => /lib64/libexpat.so.1 (0x00007f8978be9000)
        libc.so.6 => /lib64/libc.so.6 (0x00007f8978855000)
        /lib64/ld-linux-x86-64.so.2 (0x0000003569c00000)
        libXau.so.6 => /usr/lib64/libXau.so.6 (0x00007f8978652000)
        librt.so.1 => /lib64/librt.so.1 (0x00007f8978449000)
    [simon@localhost egl]$ 

        [simon@localhost egl]$ l /usr/lib/*nvidia*
    lrwxrwxrwx. 1 root root       23 Mar  9  2017 /usr/lib/libEGL_nvidia.so.0 -> libEGL_nvidia.so.367.48
    -rwxr-xr-x. 1 root root   670872 Mar  9  2017 /usr/lib/libEGL_nvidia.so.367.48
    lrwxrwxrwx. 1 root root       29 Mar  9  2017 /usr/lib/libGLESv1_CM_nvidia.so.1 -> libGLESv1_CM_nvidia.so.367.48
    -rwxr-xr-x. 1 root root    46808 Mar  9  2017 /usr/lib/libGLESv1_CM_nvidia.so.367.48
    lrwxrwxrwx. 1 root root       26 Mar  9  2017 /usr/lib/libGLESv2_nvidia.so.2 -> libGLESv2_nvidia.so.367.48
    -rwxr-xr-x. 1 root root    71384 Mar  9  2017 /usr/lib/libGLESv2_nvidia.so.367.48
    ...

    [simon@localhost egl]$ l /usr/lib64/*nvidia*
    lrwxrwxrwx. 1 root root       23 Mar  9  2017 /usr/lib64/libEGL_nvidia.so.0 -> libEGL_nvidia.so.367.48
    lrwxrwxrwx. 1 root root       29 Mar  9  2017 /usr/lib64/libGLESv1_CM_nvidia.so.1 -> libGLESv1_CM_nvidia.so.367.48
    lrwxrwxrwx. 1 root root       26 Mar  9  2017 /usr/lib64/libGLESv2_nvidia.so.2 -> libGLESv2_nvidia.so.367.48
    lrwxrwxrwx. 1 root root       21 Mar  9  2017 /usr/lib64/libnvidia-encode.so -> libnvidia-encode.so.1
    ...



* https://devtalk.nvidia.com/default/topic/670178/?comment=4114742

This log suggestes all them libs coming from nvidia install, or maybe
that they are being backed up ? 
Hmm perhaps a subsequent install replaced libs ? Seems unlikely.

::

    [simon@localhost egl]$ sudo grep lib64 /var/lib/nvidia/log
    100: /usr/lib64/libEGL.so.1.0.0
    2: /usr/lib64/libGL.so.1
    101: /usr/lib64/xorg/modules/libglamoregl.so
    102: /usr/lib64/xorg/modules/extensions/libglx.so
    2: /usr/lib64/libEGL.so.1
    103: /usr/lib64/libGL.so.1.2.0
    1: /usr/lib64/libnvidia-glcore.so.367.48
    1: /usr/lib64/xorg/modules/extensions/libglx.so.367.48
    1: /usr/lib64/libnvidia-tls.so.367.48
    1: /usr/lib64/tls/libnvidia-tls.so.367.48
    1: /usr/lib64/libGLX_nvidia.so.367.48
    1: /usr/lib64/libOpenGL.so.0
    1: /usr/lib64/libGLESv1_CM.so.1
    1: /usr/lib64/libGLESv2.so.2
    1: /usr/lib64/libGLdispatch.so.0
    1: /usr/lib64/libGLX.so.0
    1: /usr/lib64/libGL.so.1.0.0
    1: /usr/lib64/libEGL.so.1
    1: /usr/lib64/xorg/modules/drivers/nvidia_drv.so
    1: /usr/lib64/xorg/modules/libnvidia-wfb.so.367.48
    1: /usr/lib64/libnvidia-gtk2.so.367.48
    1: /usr/lib64/libnvidia-gtk3.so.367.48
    1: /usr/lib64/libnvidia-cfg.so.367.48


Note related warning::

    vi /var/log/nvidia-installer.log  

    488 -> done.
    489 -> Post-install sanity check passed.
    490 -> Running runtime sanity check:
    491 WARNING: Unable to perform the runtime configuration check for 32-bit library 'libEGL.so.1' ('/usr/lib/libEGL.so.1'); this is typically caused by the lack of a 32-bit compatibility environment.  Assum    ing successful installation.
    492 -> done.
    493 -> Runtime sanity check passed.
    494 -> Installation of the kernel module for the NVIDIA Accelerated Graphics Driver for Linux-x86_64 (version 367.48) is now complete.



* https://github.com/NVIDIA/nvidia-installer
* https://github.com/NVIDIA/nvidia-installer/issues/1


EGL without X on TK1

https://devtalk.nvidia.com/default/topic/786590/?comment=4352915


http://us.download.nvidia.com/XFree86/Linux-x86/375.26/README/installedcomponents.html

Vendor neutral graphics libraries provided by libglvnd
(/usr/lib/libOpenGL.so.0, /usr/lib/libGLX.so.0, and
/usr/lib/libGLdispatch.so.0); these libraries are currently used to provide
full OpenGL dispatching support to NVIDIA's implementation of EGL.

Source code for libglvnd is available at https://github.com/NVIDIA/libglvnd




EOU
}
egl-dir(){ echo $(env-home)/graphics/egl ; }
egl-cd(){  cd $(egl-dir); }


egl--()
{
    egl-cd


    g++ myfirst.cc -lEGL -lGL && ./a.out && ldd a.out && rm a.out 
    #g++ myfirst.cc  -L/usr/lib64 -lEGL_nvidia   -lGL && ./a.out && ldd a.out && rm a.out 


}


