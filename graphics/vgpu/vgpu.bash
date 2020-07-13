# === func-gen- : graphics/vgpu/vgpu fgp graphics/vgpu/vgpu.bash fgn vgpu fgh graphics/vgpu src base/func.bash
vgpu-source(){   echo ${BASH_SOURCE} ; }
vgpu-edir(){ echo $(dirname $(vgpu-source)) ; }
vgpu-ecd(){  cd $(vgpu-edir); }
vgpu-dir(){  echo $LOCAL_BASE/env/graphics/vgpu/vgpu ; }
vgpu-cd(){   cd $(vgpu-dir); }
vgpu-vi(){   vi $(vgpu-source) ; }
vgpu-env(){  elocal- ; }
vgpu-usage(){ cat << EOU

Virtual GPU : in generic sense, not just NVIDIA vGPU
========================================================


Old NVIDIA document
---------------------

* https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/remote-viz-tesla-gpus.pdf


See Also
----------

* nvgpu-  


Cloud Providers
------------------

AWS
~~~~

* https://rse.shef.ac.uk/blog/2020-03-24-configuring-cuda-and-opengl-courses-in-the-cloud-md/
* https://www.instancehub.com/

Google
~~~~~~~~

* https://cloud.google.com/solutions/creating-a-virtual-gpu-accelerated-linux-workstation
* https://www.teradici.com/what-is-pcoip

Oracle
~~~~~~~~

* https://blogs.oracle.com/cloud-infrastructure/using-opengl-to-enhance-gpu-use-cases-on-oracle-cloud


TurboVNC
----------

* https://www.turbovnc.org
* https://turbovnc.org/About/Introduction

All VNC implementations, including TurboVNC, use the RFB (remote framebuffer)
protocol to send “framebuffer updates” from the VNC server to any connected
"viewers." Each framebuffer update can contain multiple "rectangles" (regions
that have changed since the last update.)


VNC : Virtual Network Computing

* https://tigervnc.org
* http://libvnc.github.io  (GPL)



Guide : TurboVNC + VirtualGL
-----------------------------

* :google:`virtualgl turbovnc tutorial` 

* https://gist.github.com/cyberang3l/422a77a47bdc15a0824d5cca47e64ba2

* https://kitware.github.io/paraviewweb/docs/virtualgl_turbovnc_howto.html


CentOS7 Shut Down Display Manager ?
-------------------------------------

::

    [blyth@localhost ~]$ ps aux | grep gdm
    root      13771  0.0  0.0 481320  6660 ?        Ssl  Jun03   0:00 /usr/sbin/gdm
    root      13811  0.0  0.0 288228 44572 tty1     Ssl+ Jun03   0:18 /usr/bin/X :0 -background none -noreset -audit 4 -verbose -auth /run/gdm/auth-for-gdm-TUTuhr/database -seat seat0 -nolisten tcp vt1
    root      15065  0.0  0.0 533416  5764 ?        Sl   Jun03   0:00 gdm-session-worker [pam/gdm-password]
    root      95769  0.0  0.0 383708  4996 ?        Sl   Jun04   0:00 gdm-session-worker [pam/gdm-password]
    blyth    142510  0.0  0.0 112712   980 pts/1    S+   03:57   0:00 grep --color=auto gdm
    root     441838  0.0  0.0 383708  6780 ?        Sl   Jun12   0:00 gdm-session-worker [pam/gdm-password]
    [blyth@localhost ~]$ 


    service gdm status
    service gdm stop



VirtualGL
----------

::

    [blyth@localhost ~]$ yum info VirtualGL
    ...
    Name        : VirtualGL
    Arch        : x86_64
    Version     : 2.5.2
    Release     : 1.el7
    Size        : 562 k
    Repo        : epel/x86_64
    Summary     : A toolkit for displaying OpenGL applications to thin clients
    URL         : http://www.virtualgl.org/
    License     : wxWindows
    Description : VirtualGL is a toolkit that allows most Unix/Linux OpenGL applications to be
                : remotely displayed with hardware 3D acceleration to thin clients, regardless
                : of whether the clients have 3D capabilities, and regardless of the size of the
                : 3D data being rendered or the speed of the network.
                : 
                : Using the vglrun script, the VirtualGL "faker" is loaded into an OpenGL
                : application at run time.  The faker then intercepts a handful of GLX calls,
                : which it reroutes to the server's X display (the "3D X Server", which
                : presumably has a 3D accelerator attached.)  The GLX commands are also
                : dynamically modified such that all rendering is redirected into a Pbuffer
                : instead of a window.  As each frame is rendered by the application, the faker
                : reads back the pixels from the 3D accelerator and sends them to the
                : "2D X Server" for compositing into the appropriate X Window.
                : 
                : VirtualGL can be used to give hardware-accelerated 3D capabilities to VNC or
                : other X proxies that either lack OpenGL support or provide it through software
                : rendering.  In a LAN environment, VGL can also be used with its built-in
                : high-performance image transport, which sends the rendered 3D images to a
                : remote client (vglclient) for compositing on a remote X server.  VirtualGL
                : also supports image transport plugins, allowing the rendered 3D images to be
                : sent or captured using other mechanisms.
                : 
                : VirtualGL is based upon ideas presented in various academic papers on
                : this topic, including "A Generic Solution for Hardware-Accelerated Remote
                : Visualization" (Stegmaier, Magallon, Ertl 2002) and "A Framework for
                : Interactive Hardware Accelerated Remote 3D-Visualization" (Engel, Sommer,
                : Ertl 2000.)

    [blyth@localhost ~]$ 





EOU
}
vgpu-get(){
   local dir=$(dirname $(vgpu-dir)) &&  mkdir -p $dir && cd $dir

}
