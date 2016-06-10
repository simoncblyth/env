opticks-failed-build-vi(){ vi $BASH_SOURCE ; }
opticks-failed-build-usage(){ cat << \EOU

Opticks Failed Builds
============================

Failed Builds:

* G4PB : old OSX runs into glfw incompatibility, potentially OpenGL version problems too
* G5 : headless Linux, remote viz over X11 


G4PB build
-----------

* /usr/local/opticks : permission denied


glfw 3.1.1 needs OSX 10.6+ ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* TODO: examine glfw history to find older version prior to 10.6 isms



G5 build
----------

* glfw:CMake 2.8.9 or higher is required.  You are running version 2.6.4


runtime path problem
~~~~~~~~~~~~~~~~~~~~~~

::

    -- Up-to-date: /home/blyth/local/opticks/gl/tex/vert.glsl
    -- Installing: /home/blyth/local/opticks/lib/libGGeoViewLib.so
    -- Set runtime path of "/home/blyth/local/opticks/lib/libGGeoViewLib.so" to ""
    -- Installing: /home/blyth/local/opticks/bin/GGeoView
    -- Set runtime path of "/home/blyth/local/opticks/bin/GGeoView" to ""
    -- Up-to-date: /home/blyth/local/opticks/include/GGeoView/App.hh

* currently kludged via LD_LIBRARY_PATH


NPY/jsonutil.cpp boost::ptree compilation warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maybe split off ptree dependency togther with BRegex or BCfg


remote running : X11 failed to connect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you do not ssh in with -Y or X11 forwarding has been disabled you will get::

    X11: Failed to open X display


X11 : headless end of the line ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:env blyth$ ssh -Y G5
    Enter passphrase for key '/Users/blyth/.ssh/id_rsa': 
    Last login: Wed Apr 27 14:17:19 2016 from simon.phys.ntu.edu.tw
    [blyth@ntugrid5 ~]$ opticks-
    [blyth@ntugrid5 ~]$ opticks-check
    [2016-04-27 18:02:59.379580] [0x00002b55c3eaafe0] [info]    Opticks::preargs argc 2 argv[0] /home/blyth/local/opticks/bin/GGeoView mode Interop
    ...
    [2016-04-27 18:02:59.389754]:info: App::prepareViz size 2880,1704,2,0 position 200,200,0,0
    X11: RandR gamma ramp support seems brokenGLX: Failed to create context: GLXBadFBConfig
    [blyth@ntugrid5 ~]$ 


* reducing size to 1024,768,1 made no difference
* also running from an X11 terminal made no difference

* http://inviwo.org/svn/inviwo/modules/glfw/ext/glfw/src/x11_gamma.c

EOU
}
