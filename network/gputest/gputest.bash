# === func-gen- : network/gputest/gputest fgp network/gputest/gputest.bash fgn gputest fgh network/gputest
gputest-src(){      echo network/gputest/gputest.bash ; }
gputest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gputest-src)} ; }
gputest-vi(){       vi $(gputest-source) ; }
gputest-env(){      elocal- ; }
gputest-usage(){ cat << EOU

GPUTEST at IHEP
=================

XServer setup etc..
--------------------

* https://pleiades.ucsc.edu/hyades/OpenGL_on_Nvidia_K20



Error when don't specify CUDA_VISIBLE_DEVICE
--------------------------------------------


::

    [2015-Jul-23 15:26:36.132392]: OptiXEngine::preprocess
    [2015-Jul-23 15:26:36.132541]: OptiXEngine::preprocess start validate 
    [2015-Jul-23 15:26:36.132625]: OptiXEngine::preprocess start compile 
    [2015-Jul-23 15:26:36.288857]: OptiXEngine::preprocess start building Accel structure 
    terminate called after throwing an instance of 'optix::Exception'
      what():  Invalid value (Details: Function "RTresult _rtContextLaunch1D(RTcontext, unsigned int, RTsize)" caught exception: GL error: Invalid enum
      , [10420284])



Access
-------

* ssh lxslc509.ihep.ac.cn
* ssh gputest.ihep.ac.cn

Start TurboVNC::

   turbovnc-viewer

* in *Options > Security* set:
 
  * SSH user:blyth
  * SSH host:lxslc509.ihep.ac.cn
  * leave "Use VNC server as gateway" unchecked

* in main little window set eg:

  * VNC server:gputest.ihep.ac.cn:2
  * The port ":2" is listed on starting the vncserver on the target node

* NB TurboVNC operates with a single client per vncserver, 
  so need to to ssh in and start the vncserver first, and 
  use the port eg ":2" listed in the connection dialog above  

* to run OpenGL apps via VirtualGL prefix with "vglrun"



Start VNCServer
----------------

::
 
   /opt/TurboVNC/bin/vncserver
   Xorg


::

    -bash-4.1$ /opt/TurboVNC/bin/vncserver :5

    You will require a password to access your desktops.

    Password: 
    Warning: password truncated to the length of 8.
    Verify:   
    Would you like to enter a view-only password (y/n)? n

    Desktop 'TurboVNC: gputest.ihep.ac.cn:5 (blyth)' started on display gputest.ihep.ac.cn:5

    Creating default startup script /afs/ihep.ac.cn/users/b/blyth/.vnc/xstartup.turbovnc
    Starting applications specified in /afs/ihep.ac.cn/users/b/blyth/.vnc/xstartup.turbovnc
    Log file is /afs/ihep.ac.cn/users/b/blyth/.vnc/gputest.ihep.ac.cn:5.log


Normal Start
-------------
::

    -bash-4.1$ /opt/TurboVNC/bin/vncserver :5

    Desktop 'TurboVNC: gputest.ihep.ac.cn:5 (blyth)' started on display gputest.ihep.ac.cn:5

    Starting applications specified in /afs/ihep.ac.cn/users/b/blyth/.vnc/xstartup.turbovnc
    Log file is /afs/ihep.ac.cn/users/b/blyth/.vnc/gputest.ihep.ac.cn:5.log


    -bash-4.1$ 
    -bash-4.1$ 
    -bash-4.1$ Xorg :6

    X.Org X Server 1.15.0
    Release Date: 2013-12-27
    X Protocol Version 11, Revision 0
    Build Operating System: sl6 2.6.32-504.el6.x86_64 
    Current Operating System: Linux gputest.ihep.ac.cn 2.6.32-504.3.3.el6.x86_64 #1 SMP Tue Dec 16 14:29:22 CST 2014 x86_64
    ...


From Virtual Terminal
-----------------------

Using the X window number of Xorg above::

   bash-4.1$ vglrun -d :6 glxgears

AFS
---

::

    ssh lxslc6.ihep.ac.cn
    cd  /publicfs/dyb/user/blyth


Kill VNCServer
----------------

Cleanup::

    -bash-4.1$ /opt/TurboVNC/bin/vncserver -kill :5
    Killing Xvnc process ID 85725

Start Xorg from non-virtual terminal
--------------------------------------

::

    -bash-4.1$ Xorg :6


Kill Xorg
------------

::

    -bash-4.1$ ps aux | grep Xorg
    root      22444  0.1  0.0 156884 56956 tty8     Ss+  12:45   0:18 Xorg
    root      87092  0.4  0.0 153052 53068 tty9     Ss+  16:46   0:02 Xorg :6
    blyth     89686  0.0  0.0 103256   872 pts/7    S+   16:54   0:00 grep Xorg
    -bash-4.1$ kill 87092
    -bash-4.1$ ps aux | grep Xorg
    root      22444  0.1  0.0 156884 56956 tty8     Ss+  12:45   0:18 Xorg
    blyth     89793  0.0  0.0 103256   872 pts/7    S+   16:55   0:00 grep Xorg
    -bash-4.1$ 




X window number for VirtualGL is distinct from the VNC server number
----------------------------------------------------------------------

::

    vglrun -d :6 glxgears








Linux OptiX 3.8.0 SDK Samples
---------------------------------

::

    /afs/ihep.ac.cn/soft/juno/JUNO-ALL-SLC6/GPU/20150723/OptiX


EOU
}
gputest-dir(){ echo $(local-base)/env/network/gputest/network/gputest-gputest ; }
gputest-cd(){  cd $(gputest-dir); }
gputest-mate(){ mate $(gputest-dir) ; }
gputest-get(){
   local dir=$(dirname $(gputest-dir)) &&  mkdir -p $dir && cd $dir

}
