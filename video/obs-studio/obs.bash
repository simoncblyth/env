# === func-gen- : video/obs-studio/obs fgp video/obs-studio/obs.bash fgn obs fgh video/obs-studio
obs-src(){      echo video/obs-studio/obs.bash ; }
obs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(obs-src)} ; }
obs-vi(){       vi $(obs-source) ; }
obs-env(){      elocal- ; }
obs-usage(){ cat << EOU

OBS Studio : Open Broadcaster Software
=========================================

OBS Studio is free and open source software for video recording and live streaming.

* https://obsproject.com
* https://obsproject.com/help/
* https://obsproject.com/wiki/
* https://obsproject.com/forum/search/

* paid alternatives : Camtasia, Screenflow


Following rpmfusion install of vlc (and the ffmpeg-libs, x264-libs that come with it)
--------------------------------------------------------------------------------------

* find that need to obs-export (setup LD_LIBRARY_PATH) to find obs libs

Hmm maybe can get obs from rpmfusion too ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/obsproject/obs-studio/wiki/Install-Instructions#linux

With some translation based on https://rpmfusion.org/Configuration

::

    sudo yum install https://download1.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %centos).noarch.rpm 
    sudo yum install https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-$(rpm -E %centos).noarch.rpm

    ## switched 
    ##     %fedora -> %centos
    ##     fedora->el  
 
    rpm -E %centos  ## evaluates the expression giving "7" 

    sudo yum install obs-studio
       ## failed to find this

    sudo yum install xorg-x11-drv-nvidia-cuda



rpmfusion
-----------

* https://rpmfusion.org/Configuration

* https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm
* https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-7.noarch.rpm




Usage Guides for screen recording
-------------------------------------

* https://photography.tutsplus.com/tutorials/obs-for-screen-recording-quick-start--cms-28549

Before you can start recording you'll also need to add a source inside this
scene. With the default Scene selected (it will be highlighted) click the +
button at the bottom of the panel labeled *Sources*, then select Screen Capture
on Linux or Display Capture on Mac and Windows.

* https://photography.tutsplus.com/tutorials/obs-for-screen-recording-video-and-output-settings--cms-28542


Linux runtime, many "QXcbConnection: XCB error 8..." when mousing around over the OBS window
----------------------------------------------------------------------------------------------

::

    QXcbConnection: XCB error: 8 (BadMatch), sequence: 19527, resource id: 54794349, major code: 130 (Unknown), minor code: 3
    QXcbConnection: XCB error: 8 (BadMatch), sequence: 31741, resource id: 54795203, major code: 130 (Unknown), minor code: 3
    QXcbConnection: XCB error: 8 (BadMatch), sequence: 31772, resource id: 54795203, major code: 130 (Unknown), minor code: 3

* https://forum.qt.io/topic/86643/qxcbconnection-xcb-error-8-badmatch

Claims its just a deprecation warning.


Precise Screen Recording (Linux) 
------------------------------------

1. ~/local/env/bin/obs
2. add a Source : Screen Capture (XSHM)
3.(yes:this works) right click on the source, select "Filters", "+" to add one, choose "Crop/Pad", 
  deselect "Relative" for absolute positioning, enter x:100 y:100 w:1920 h:1080 and "Close" 

* this filter is persisted between OBS runs 

::

   OKTest --size 1920,1080,1 --position 100,100 


Unlock the OBS GUI so can place widgets around the captured portion
---------------------------------------------------------------------

* https://obsproject.com/blog/whats-new-in-obs-studio-20-0

* View > Docks > "Lock UI" (uncheck)
* then can reposition the widgets, avoiding the captured portion
* then check "Lock UI" 

* this setup (breakout+place GUI windows) is persisted, so long as exit OBS cleanly  


macOS
-------

* https://github.com/obsproject/obs-studio/releases/download/21.1.1/obs-mac-21.1.1-installer.pkg
* GUI installer, states will take 354.7 MB, lands into /Applications/OBS.app

Wizard
~~~~~~~

* pick optimize for recording (not streaming)
* Base (canvas) resolution options::

  Use Current (1440x900)  [default]
  Display 1 (1440x900)
  1920x1080    << picked this << 
  1280x720

* FPS : Either 60 or 30, prefer 60 when possible


The wizard ran some tests, and concluded::

   Recording Encoder          : software (x264)
   Recording Quality          : High Quality, Medium File size
   Base (Canvas) Resolution   : 1920x1080 
   Output (Scaled) Resolution : 1280x720 
                         FPS  :  60 

In "Output" can pick Recording format::

   flv  (default)
   mp4
   mov  (picked)
   mkv
   ts 
   m3u8

  
::

    In [5]: 1440./900.   Out[5]: 1.6 
    In [1]: 1280./720.   Out[1]: 1.7777777777777777 
    In [4]: 1920./1080.  Out[4]: 1.7777777777777777 
    In [3]: 16./9.       Out[3]: 1.7777777777777777 
     


Linux
-------

* https://github.com/obsproject/obs-studio/wiki/Install-Instructions#linux
* https://github.com/obsproject/obs-studio/wiki/Install-Instructions#linux-build-directions

::

   yum info gcc gcc-c++ gcc-objc cmake3 git   # only need objc

   yum info libX11-devel          ## 1.6.5 already (base)
   yum info mesa-libGL-devel      ## 17.2.3 already (base)
   yum info libv4l-devel          ## 0.9.5 installed from base
   yum info pulseaudio-libs-devel ## 10.0 from base  (glib2-devel came with it)

   yum info x264-devel            ## not in repo, but have already manually installed with x264- as ffmpeg- dependency  

   yum info freetype-devel        ## 2.4.11 already (base)
   yum info fontconfig-devel      ## 2.10.95 installed from base
   yum info libXcomposite-devel   ## 0.4.4 installed from base 
   yum info libXinerama-devel     ## 1.1.3 already (base)
   yum info qt5-qtbase-devel      ## 5.9.2 installed from base
   yum info qt5-qtx11extras-devel ## 5.9.2 installed from base  
   yum info libcurl-devel         ## 7.29.0 installed from base
   yum info systemd-devel         ## 219 installed from base 

   yum info ffmpeg                ## not in repo, but have already manually installed with ffmpeg- 


Linux Wizard setup
~~~~~~~~~~~~~~~~~~~~~

* optimize for recording, not streaming

Base (canvas) resolution::

   Use Current (1920x1080)
   Display 1 (2560x1440)
   1920x1080
   1280x720

Wizard determined:

   Recording Encoder          :  Software (x264)
   Recording quality          :  High Quality, Medium file size
   Base (canvas) resolution   : 1920x1080
   Output (scaled) resolution : 1920x1080
   FPS                        : 60 














EOU
}
obs-prefix(){ echo $(local-base)/env ; }
obs-dir(){    echo $(local-base)/env/video/obs-studio ; }
obs-bdir(){   echo $(local-base)/env/video/obs-studio.build ; }
obs-cd(){     cd $(obs-dir); }
obs-bcd(){    cd $(obs-bdir); }

obs-get-notes(){ cat << EON

Quite a few plugins pulled by the recursive clone
/home/blyth/local/env/video/obs-studio/obs-get.log

EON
}


obs-get(){
   local dir=$(dirname $(obs-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d obs-studio ] && git clone --recursive https://github.com/obsproject/obs-studio.git
}


obs-configure-notes(){ cat << EON

/home/blyth/local/env/video/obs-studio.build/obs-configure.log

* warning from OpenGL 
* error from Qt : picking up some anaconda Qt stuff 

* eliminate anaconda from PATH avoids that confusion but still miss /usr/lib64/libEGL.so
  that is expected by Qt5::Gui 

*  yum whatprovides /usr/lib64/libEGL.so  -> mesa-libEGL-devel
   but thats already installed 


Looking at the libs, find a broken symbolic link::

    [blyth@localhost obs-studio.build]$ ll /usr/lib64/libE*
    -rwxr-xr-x. 1 root root  73328 Jul  5 15:15 /usr/lib64/libEGL.so.1.1.0
    lrwxrwxrwx. 1 root root     15 Jul  5 15:15 /usr/lib64/libEGL.so.1 -> libEGL.so.1.1.0

    -rwxr-xr-x. 1 root root 971648 Jul  5 15:15 /usr/lib64/libEGL_nvidia.so.396.26
    lrwxrwxrwx. 1 root root     23 Jul  5 15:15 /usr/lib64/libEGL_nvidia.so.0 -> libEGL_nvidia.so.396.26

    lrwxrwxrwx. 1 root root     15 Jul 17 19:55 /usr/lib64/libEGL.so -> libEGL.so.1.0.0    #### broken symbolic link 


Looks like the NVIDIA driver install stomps on the libEGL.so ? 

* https://obsproject.com/forum/threads/debian-jessie-error-on-compiling.27422/
* https://devtalk.nvidia.com/default/topic/973013/linux/libegl-so-1-libegl-so-375-10-difference-not-in-release-notes-docs/
* https://bbs.archlinux.org/viewtopic.php?id=178189
* https://www.mesa3d.org/relnotes/17.2.3.html
* https://www.mesa3d.org/egl.html
* https://devtalk.nvidia.com/default/topic/685409/linux/334-16-if-install-libegl-so-x-y-z-stop-build-mesa-libs/
* https://askubuntu.com/questions/616065/the-imported-target-qt5gui-references-the-file-usr-lib-x86-64-linux-gnu-li

Conflict between libEGL from NVIDIA driver and the one from mesa

* "MESA EGL CONFLICTS WITH NVIDIA EGL LIBRARIES"



::

    [blyth@localhost obs-studio.build]$ rpm -ql mesa-libEGL.x86_64
    /usr/lib64/libEGL.so.1
    /usr/lib64/libEGL.so.1.0.0

    [blyth@localhost obs-studio.build]$ rpm -ql mesa-libEGL-devel-17.2.3-8.20171019.el7.x86_64
    /usr/include/EGL
    /usr/include/EGL/egl.h
    /usr/include/EGL/eglext.h
    /usr/include/EGL/eglextchromium.h
    /usr/include/EGL/eglmesaext.h
    /usr/include/EGL/eglplatform.h
    /usr/include/KHR
    /usr/include/KHR/khrplatform.h
    /usr/lib64/libEGL.so
    /usr/lib64/pkgconfig/egl.pc



EON
}

obs-kludge-egl-notes(){ cat << EON


* this kludge allows obs-configure to complete, with some warnings about AUTOMOC


lrwxrwxrwx. 1 root root     23 Jul  5 15:15 libEGL_nvidia.so.0 -> libEGL_nvidia.so.396.26
-rwxr-xr-x. 1 root root 971648 Jul  5 15:15 libEGL_nvidia.so.396.26
lrwxrwxrwx. 1 root root     15 Jul 17 19:55 libEGL.so -> libEGL.so.1.0.0
lrwxrwxrwx. 1 root root     15 Jul  5 15:15 libEGL.so.1 -> libEGL.so.1.1.0
-rwxr-xr-x. 1 root root  73328 Jul  5 15:15 libEGL.so.1.1.0
sudo ln -svf libEGL.so.1.1.0 libEGL.so
[sudo] password for blyth: 
‘libEGL.so’ -> ‘libEGL.so.1.1.0’
lrwxrwxrwx. 1 root root     23 Jul  5 15:15 libEGL_nvidia.so.0 -> libEGL_nvidia.so.396.26
-rwxr-xr-x. 1 root root 971648 Jul  5 15:15 libEGL_nvidia.so.396.26
lrwxrwxrwx. 1 root root     15 Jul 17 21:28 libEGL.so -> libEGL.so.1.1.0
lrwxrwxrwx. 1 root root     15 Jul  5 15:15 libEGL.so.1 -> libEGL.so.1.1.0
-rwxr-xr-x. 1 root root  73328 Jul  5 15:15 libEGL.so.1.1.0
[blyth@localhost lib64]$ 

EON
}

obs-kludge-egl()
{
   type $FUNCNAME
   cd /usr/lib64
   ls -l libEGL*
   local cmd="sudo ln -svf libEGL.so.1.1.0 libEGL.so"
   echo $cmd
   eval $cmd

   ls -l libEGL*

}


obs-configure(){
   obs-cd

   local sdir=$(obs-dir)
   local bdir=$(obs-bdir)

   rm -rf $bdir   ## blast away 
   mkdir -p $bdir
 
   obs-bcd

   #local pref=GLVND
   local pref=LEGACY

   cmake \
      -DUNIX_STRUCTURE=1 \
      -DOpenGL_GL_PREFERENCE=$pref \
      -DCMAKE_INSTALL_PREFIX=$(obs-prefix) \
      $sdir
}

obs-make()
{
   obs-bcd
   make $*
}

obs-install-notes(){ cat << EON

Many undefined refs and::

    /usr/bin/ld: /home/blyth/local/env/bin/../lib/libavcodec.a(dirac_arith.o): relocation R_X86_64_32S against symbol ff_dirac_prob can not be used when making a shared object; recompile with -fPIC


Looks like need to align the compilation options used for ffmpeg- and obs-

* https://obsproject.com/forum/threads/compiling-obs-libswscale-a-recompile-with-fpic.47290/

* https://trac.ffmpeg.org/wiki/CompilationGuide/Centos
* https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu


* https://obsproject.com/forum/threads/building-ffmpeg-and-obs-to-use-nvenc-on-linux-mint.82554/#post-346537


EON
}



obs-install()
{
   obs-bcd

   obs-make -j4
   obs-make install
}

obs--()
{
   obs-get
   obs-configure
   obs-install
}


obs-export()
{
   export LD_LIBRARY_PATH=$(obs-prefix)/lib:$LD_LIBRARY_PATH 
}

