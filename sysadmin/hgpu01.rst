HGPU01
========

Quick Access
-------------

After the ssh agents are running on source node and the chosen lxslc gateway node
can get through with 2 passwordless hops::

    delta:~ blyth$ ssh L6
    -bash-3.2$ ssh G1
    -bash-4.1$ 

Forced command could be used to shrink this to one hop.


Access
-------

#. `ssh L` (this picks a random lxslc node eg lxslc506.ihep.ac.cn)  
#. `klog`  

   * annoyingly need to AFS authenticate manually after login, 
     otherwise no write access to directories

#. because are coming back to a random node, there will usually be 
   no ssh-agent running, so start it

   * `ssh--agent-start` authenticate with ssh key passphrase, not the same a AFS password 


#. now can passwordlessly access G1

   * `ssh G1`

#. and yet again need to authenticate against AFS

   * `klog`


To avoid having to start ssh agents everytime, go via the 
same gateway node using **L6**

::

    delta:~ blyth$ ssh L6
    Last login: Mon Jan 26 11:40:33 2015 from simon.phys.ntu.edu.tw
    **********************************************************************
    |  Time  |      Up Time     |Loing Users|        Load Average        |
     11:40:52 up 24 days, 13:48, 40 users,  load average: 0.06, 0.33, 0.91
    **********************************************************************
    TEL:5037(office);83050656
    -bash-3.2$ ssh G1
    Last login: Mon Jan 26 11:40:38 2015 from lxslc506.ihep.ac.cn
    -bash-4.1$ 


Mercurial
----------

Ancient Mercurial does not work with the local hg 
I installed on lxslc::

    -bash-4.1$ e
    abort: requirement 'dotencode' not supported!
    -bash-4.1$ t e
    e is aliased to `cd $ENV_HOME ; hg st '
    -bash-4.1$ hg st 
    abort: requirement 'dotencode' not supported!
    -bash-4.1$ 

::

    -bash-4.1$ which hg
    /usr/bin/hg
    -bash-4.1$ hg --version
    Mercurial Distributed SCM (version 1.4)

    Copyright (C) 2005-2009 Matt Mackall <mpm@selenic.com> and others
    This is free software; see the source for copying conditions. There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    -bash-4.1$ 



Trying to use the mercurial built on lxslc on hgpu01 dont work (accessed via afs).
Presumably as using different pythons.::

    -bash-4.1$ ~/local/env/hg/mercurial-3.2/hg status
    ...
       mod = _hgextimport(_import, head, globals, locals, None, level)
      File "/afs/ihep.ac.cn/users/b/blyth/local/env/hg/mercurial-3.2/mercurial/demandimport.py", line 47, in _hgextimport
        return importfunc(name, globals, *args)
    ImportError: /afs/ihep.ac.cn/users/b/blyth/local/env/hg/mercurial-3.2/mercurial/osutil.so: undefined symbol: Py_InitModule4
    -bash-4.1$ 


Network
--------

Its cut off fom the world::

    -bash-4.1$ ping bitbucket.org
    PING bitbucket.org (131.103.20.167) 56(84) bytes of data.


This means need to annoyingly pull in code from lxslc 


Storage
--------

::

    /dyb/dybd07/user/blyth




Headless OpenGL ?
--------------------

Following the below article I note that the graphics operation mode “GOM”
shown above is set to “Compute” only which prevents use of OpenGL 

* http://devblogs.nvidia.com/parallelforall/interactive-supercomputing-in-situ-visualization-tesla-gpus/

::

    -bash-4.1$ nvidia-smi --format=csv --query-gpu=gom.current
    gom.current
    Compute
    Compute
    -bash-4.1$ 

    -bash-4.1$ nvidia-smi -i 0 --gom=0
    Unable to set GOM to "All On" for GPU 0000:03:00.0: Insufficient Permissions
    Terminating early due to previous errors.

    -bash-4.1$ nvidia-smi -i 1 --gom=0
    Unable to set GOM to "All On" for GPU 0000:84:00.0: Insufficient Permissions
    Terminating early due to previous errors.


Enabling GOM is just the first step...
it would be necessary to run an X server on the node to provide 
OpenGL with somewhere to put its context.    I am not sure how 
to do headless OpenGL rendering but possibly using 
a virtual framebuffer like Xvfb would work ?

     http://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml

Hmm maybe simpler to output OptiX buffers into ppm or png ?


libPNG
-------

::

    -bash-4.1$ locate libpng
    /usr/bin/libpng-config
    /usr/bin/libpng12-config
    /usr/include/libpng12
    /usr/include/libpng12/png.h
    /usr/include/libpng12/pngconf.h
    /usr/lib/libpng.so
    /usr/lib/libpng.so.3
    /usr/lib/libpng.so.3.49.0
    /usr/lib/libpng12.so
    /usr/lib/libpng12.so.0
    /usr/lib/libpng12.so.0.49.0
    /usr/lib/pkgconfig/libpng.pc
    /usr/lib/pkgconfig/libpng12.pc
    /usr/lib64/libpng.so
    /usr/lib64/libpng.so.3
    ...

