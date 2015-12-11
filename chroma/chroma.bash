# === func-gen- : muon_simulation/chroma/chroma fgp muon_simulation/chroma/chroma.bash fgn chroma fgh muon_simulation/chroma
chroma-src(){      echo chroma/chroma.bash ; }
chroma-source(){   echo ${BASH_SOURCE:-$(env-home)/$(chroma-src)} ; }
chroma-vi(){       vi $(chroma-source) ; }
chroma-usage(){ cat << EOU

CHROMA
=======

* http://chroma.bitbucket.org/install/overview.html
* http://chroma.bitbucket.org/install/macosx.html
* http://chroma.bitbucket.org/install/overview.html#common-install
* https://bitbucket.org/chroma/chroma.bitbucket.org/src


Top Level Link
----------------

::

    simon:~ blyth$ l chroma
    lrwxr-xr-x  1 blyth  staff  36 Oct  9  2014 chroma -> /usr/local/env/chroma_env/src/chroma


Update for CUDA 7, Dec 2015
------------------------------

pycuda import fails for lack of a CUDA 5.5 curand lib, reinstallation of pycuda fixes this:

   chroma-
   pip install -b /usr/local/env/chroma_env/build/build_pycuda pycuda --upgrade
   rm -rf /usr/local/env/chroma_env/build/build_pycuda/pycuda/
   pip install -b /usr/local/env/chroma_env/build/build_pycuda pycuda --upgrade


Installation Overview
----------------------

* contained within virtualenv folder chroma-dir
  eg /usr/local/env/chroma_env

Dependencies Install Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After setup of ~/.aksetup-defaults.py described in http://chroma.bitbucket.org/install/overview.html::

    #export PIP_EXTRA_INDEX_URL=http://mtrr.org/chroma_pkgs/ 
    #export PIP_EXTRA_INDEX_URL=http://localhost/chroma_pkgs/ 
    export PIP_EXTRA_INDEX_URL=file:///usr/local/env/chroma_pkgs/ 
    pip install chroma_deps

Chroma dependencies installation uses homegrown *shrinkwrap* 
that makes non-python packages such as Geant4 and ROOT 
to be amenable to setup.py python package 
dependency machinery.  

Packages are pulled from server using PIP indexing 
mechanism. Recall needing to setup localhost index in order
to conveniently make some patches. 

IMPORTANT
~~~~~~~~~~~

* pip logs to ~/.pip/pip.log 
* overwritten at each call


chroma deps server
~~~~~~~~~~~~~~~~~~~~

Bring up local nginx::

    delta:~ blyth$ sudo apachectl stop
    delta:~ blyth$ nginx-start


To provide the below via http://localhost/chroma_pkgs/::

   (chroma_env)delta:Documents blyth$ ll /usr/local/env/chroma_pkgs/
    total 0
    drwxr-xr-x   4 blyth  wheel  136 Jan 16  2014 g4py_chroma
    drwxr-xr-x   8 blyth  wheel  272 Jan 16  2014 .
    drwxr-xr-x   5 blyth  wheel  170 Jan 16  2014 cmake
    drwxr-xr-x   5 blyth  wheel  170 Jan 16  2014 chroma_deps
    drwxr-xr-x   4 blyth  wheel  136 Jan 16  2014 boost
    drwxr-xr-x   7 blyth  wheel  238 Jan 16  2014 geant4
    drwxr-xr-x   8 blyth  wheel  272 Jan 17  2014 root
    drwxr-xr-x  23 blyth  staff  782 Sep 25 12:26 ..



D: modify chroma geant4 build to add GDML/persistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Macports install of xercesc, see *xercesc-vi*

Want to add GDML to Chroma geant4 build, need to change cmake cmdline::

    delta:geant4-4.9.5.p01 blyth$ pwd
    /usr/local/env/chroma_pkgs/geant4/geant4-4.9.5.p01

    delta:geant4-4.9.5.p01 blyth$ cat setup.py  | grep cmake
        config_cmd = 'cmake -DCMAKE_INSTALL_PREFIX=%s -DGEANT4_INSTALL_DATA=ON -DGEANT4_USE_RAYTRACER_X11=ON %s' % (inst.virtualenv, g4src_dir)
        shrinkwrap_requires=['cmake'],


* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/ch02.html

::

    delta:geant4-4.9.5.p01 blyth$ diff setup.py ../geant4-4.9.5.post2/setup.py 
    9,10c9,11
    < version = '4.9.5.p01'
    < source_url = 'http://geant4.cern.ch/support/source/geant%s.tar.gz' % version
    ---
    > version = '4.9.5.post2'
    > real_version = '4.9.5.p01'
    > source_url = 'http://geant4.cern.ch/support/source/geant%s.tar.gz' % real_version
    18c19
    <     g4src_dir = os.path.realpath('geant' + version)
    ---
    >     g4src_dir = os.path.realpath('geant' + real_version)
    25,26c26
    <     xopt = "-DGEANT4_USE_GDML=ON"
    <     config_cmd = 'cmake -DCMAKE_INSTALL_PREFIX=%s -DGEANT4_INSTALL_DATA=ON -DGEANT4_USE_RAYTRACER_X11=ON %s %s' % (inst.virtualenv, xopt, g4src_dir)
    ---
    >     config_cmd = 'cmake -DCMAKE_INSTALL_PREFIX=%s -DGEANT4_INSTALL_DATA=ON -DGEANT4_USE_RAYTRACER_X11=ON %s' % (inst.virtualenv, g4src_dir)
    46c46
    <     shrinkwrap_requires=['cmake'],
    ---
    >    # shrinkwrap_requires=['cmake'],
    delta:geant4-4.9.5.p01 blyth$ 


Need to add::

    (chroma_env)delta:geant4 blyth$ vi geant4-4.9.5.post2/setup.py   ## add -DGEANT_USE_GDML=ON
    (chroma_env)delta:geant4 blyth$ rm  geant4-4.9.5.post2.tar.gz
    (chroma_env)delta:geant4 blyth$ tar zcvf geant4-4.9.5.post2.tar.gz geant4-4.9.5.post2
    a geant4-4.9.5.post2
    a geant4-4.9.5.post2/geant4.egg-info
    a geant4-4.9.5.post2/PKG-INFO
    a geant4-4.9.5.post2/setup.cfg
    a geant4-4.9.5.post2/setup.py
    a geant4-4.9.5.post2/geant4.egg-info/dependency_links.txt
    a geant4-4.9.5.post2/geant4.egg-info/PKG-INFO
    a geant4-4.9.5.post2/geant4.egg-info/SOURCES.txt
    a geant4-4.9.5.post2/geant4.egg-info/top_level.txt




the pip install sources geant4.sh::

    delta:chroma_env blyth$ find . -name geant4.sh
    ./bin/geant4.sh
    ./env.d/geant4.sh
    ./src/geant4.9.5.p01-build/InstallTreeFiles/geant4.sh

    delta:chroma_env blyth$ cat env.d/geant4.sh 

    pushd .  > /dev/null
    cd /usr/local/env/chroma_env/bin
    source geant4.sh
    popd  > /dev/null


Add tail::

    delta:chroma_env blyth$ tail -10 bin/geant4.sh

    # SCB extra
    xercesc_library=/opt/local/lib/libxerces-c.dylib 
    if [ -f "${xercesc_library}" ]; then 
       export XERCESC_INCLUDE_DIR=/opt/local/include
       export XERCESC_LIBRARY=/opt/local/lib/libxerces-c.dylib 
       env | grep XERCESC
    fi


 
Remove that tail, try with env setup in chroma-deps-env, then::

     (chroma_env)delta:chroma_env blyth$ chroma-deps-rebuild geant4 -U -v


::

    (chroma_env)delta:chroma_env blyth$ chroma-;chroma-deps-rebuild-geant4 -U -v --pre
    === chroma-deps-env : writing /Users/blyth/.aksetup-defaults.py
    XERCESC_INCLUDE_DIR=/opt/local/include
    XERCESC_LIBRARY=/opt/local/lib/libxerces-c.dylib
    === chroma-deps-rebuild : pip install -b /usr/local/env/chroma_env/build/build_geant4 -U -v --pre geant4
    Could not fetch URL https://pypi.python.org/simple/geant4/: HTTP Error 404: Not Found
    Will skip URL https://pypi.python.org/simple/geant4/ when looking for download links for geant4 in ./lib/python2.7/site-packages
    Could not fetch URL https://pypi.python.org/simple/geant4/: HTTP Error 404: Not Found
    Will skip URL https://pypi.python.org/simple/geant4/ when looking for download links for geant4 in ./lib/python2.7/site-packages
    Installed version (4.9.5.post2) is most up-to-date (past versions: 4.9.5.post2, 4.9.5.post1, 4.9.5.p01)
    Requirement already up-to-date: geant4 in ./lib/python2.7/site-packages
    Cleaning up...
    (chroma_env)delta:chroma_env blyth$ 


marker directory in site-packages::

    (chroma_env)delta:site-packages blyth$ l geant4-4.9.5.post2-py2.7.egg-info/
    total 40
    -rw-r--r--  1 blyth  staff  197 Jan 16  2014 PKG-INFO
    -rw-r--r--  1 blyth  staff  130 Jan 16  2014 SOURCES.txt
    -rw-r--r--  1 blyth  staff    1 Jan 16  2014 dependency_links.txt
    -rw-r--r--  1 blyth  staff   59 Jan 16  2014 installed-files.txt
    -rw-r--r--  1 blyth  staff    1 Jan 16  2014 top_level.txt
    (chroma_env)delta:site-packages blyth$ 

Need to remove 3 dirs to succeeds to rebuild::

    (chroma_env)delta:chroma_env blyth$ rm -r /usr/local/env/chroma_env/build/build_geant4
    (chroma_env)delta:chroma_env blyth$ rm -r lib/python2.7/site-packages/geant4-4.9.5.post2-py2.7.egg-info
    (chroma_env)delta:chroma_env blyth$ rm -rf /usr/local/env/chroma_env/src/geant4.9.5.p01   
        # otherwise get permission denied regarding examples/.../Doxyfile.svn-base

    (chroma_env)delta:chroma_env blyth$ chroma-;chroma-deps-rebuild-geant4 -U -v --pre
     
But my change to geant4-4.9.5.post2/setup.py was ignored. Need to create tarball::

    (chroma_env)delta:geant4 blyth$ pwd
    /usr/local/env/chroma_pkgs/geant4
    (chroma_env)delta:geant4 blyth$ rm geant4-4.9.5.post2.tar.gz
    (chroma_env)delta:geant4 blyth$ tar zcvf geant4-4.9.5.post2.tar.gz geant4-4.9.5.post2
    a geant4-4.9.5.post2
    a geant4-4.9.5.post2/geant4.egg-info
    a geant4-4.9.5.post2/PKG-INFO
    a geant4-4.9.5.post2/setup.cfg
    a geant4-4.9.5.post2/setup.py
    a geant4-4.9.5.post2/geant4.egg-info/dependency_links.txt
    a geant4-4.9.5.post2/geant4.egg-info/PKG-INFO
    a geant4-4.9.5.post2/geant4.egg-info/SOURCES.txt
    a geant4-4.9.5.post2/geant4.egg-info/top_level.txt
    (chroma_env)delta:geant4 blyth$ 


After that succeed to rebuild but getting::

   CMake Warning:
      Manually-specified variables were not used by the project:
    
        GEANT_USE_GDML
    




Chroma Dependencies
--------------------

OSX 10.9.1 Mavericks 
~~~~~~~~~~~~~~~~~~~~~~

* Xcode 5.0.2 with commandline tools
* XQuartz 2.7.5
* CUDA 5.5
* Macports (logs in ~/macports/)

  * py27-matplotlib 
  * mercurial 
  * py27-game 
  * py27-virtualenv 
  * Xft2 
  * xpm

Common
~~~~~~~~

* http://shrinkwrap.readthedocs.org/en/latest/

Careful with errant eggs 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After testing build_ext::

    (chroma_env)delta:chroma blyth$ touch src/G4chroma.cc
    (chroma_env)delta:chroma blyth$ python setup.py build_ext
    running build_ext
    building 'chroma.generator._g4chroma' extension
    C compiler: /usr/bin/clang -fno-strict-aliasing -fno-common -dynamic -pipe -Os -fwrapv -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes

    compile options: '-Isrc -I/usr/local/env/chroma_env/lib/python2.7/site-packages/pyublas/include -I/usr/local/env/chroma_env/include -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -c'
    extra options: '-DG4INTY_USE_XT -I/usr/X11R6/include -I/opt/local/include -DG4UI_USE_TCSH -DG4VIS_USE_RAYTRACERX -I/usr/local/env/chroma_env/bin/../include/Geant4'
    clang: src/G4chroma.cc
    ...
    /usr/bin/clang++ -bundle -undefined dynamic_lookup -L/opt/local/lib -Wl,-headerpad_max_install_names -L/opt/local/lib/db46 build/temp.macosx-10.9-x86_64-2.7/src/G4chroma.o -lboost_python -o build/lib.macosx-10.9-x86_64-2.7/chroma/generator/_g4chroma.so -L/usr/local/env/chroma_env/bin/../lib -lG4Tree -lG4FR -lG4GMocren -lG4visHepRep -lG4RayTracer -lG4VRML -lG4vis_management -lG4modeling -lG4interfaces -lG4persistency -lG4analysis -lG4error_propagation -lG4readout -lG4physicslists -lG4run -lG4event -lG4tracking -lG4parmodels -lG4processes -lG4digits_hits -lG4track -lG4particles -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms -lG4global -lG4clhep
    (chroma_env)delta:chroma blyth$ 

Find launch fails of g4daeview.py, with indication of looking inside an errant egg::

   self._gpu_geometry = self.setup_gpu_geometry()
  File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daechromacontext.py", line 73, in setup_gpu_geometry
    return GPUGeometry( self.chroma_geometry )
  File "/usr/local/env/chroma_env/lib/python2.7/site-packages/Chroma-0.5-py2.7-macosx-10.9-x86_64.egg/chroma/gpu/geometry.py", line 25, in __init__
    geometry_source = get_cu_source('geometry_types.h')
  File "<string>", line 2, in get_cu_source
  File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pytools/__init__.py", line 382, in memoize
    result = func(*args)
  File "/usr/local/env/chroma_env/lib/python2.7/site-packages/Chroma-0.5-py2.7-macosx-10.9-x86_64.egg/chroma/gpu/tools.py", line 89, in get_cu_source
    with open('%s/%s' % (srcdir, name)) as f:
  IOError: [Errno 2] No such file or directory: '/usr/local/env/chroma_env/lib/python2.7/site-packages/Chroma-0.5-py2.7-macosx-10.9-x86_64.egg/chroma/cuda/geometry_types.h'


::

    delta:~ blyth$ ll /usr/local/env/chroma_env/lib/python2.7/site-packages/Chroma-0.5-py2.7-macosx-10.9-x86_64.egg/chroma/cuda/
    total 16
    -rw-r--r--   1 blyth  staff   329 Sep 19 20:27 __init__.pyc
    -rw-r--r--   1 blyth  staff    67 Sep 19 20:27 __init__.py
    drwxr-xr-x  53 blyth  staff  1802 Sep 19 20:27 ..
    drwxr-xr-x   4 blyth  staff   136 Sep 19 20:27 .
    delta:~ blyth$ date
    Mon Sep 22 16:16:44 CST 2014

The below all failed to place the headers in the egg::

    (chroma_env)delta:chroma blyth$ python setup.py install_headers
    (chroma_env)delta:chroma blyth$ python setup.py develop
    (chroma_env)delta:chroma blyth$ python setup.py install


Recall that **develop** is the chroma setup.py command being used, so there should
be no egg. Removing the egg succeeds to resume operation::

    delta:site-packages blyth$ cat easy-install.pth 
    import sys; sys.__plen = len(sys.path)
    ./unittest2-0.5.1-py2.7.egg
    ./Sphinx-1.2-py2.7.egg
    ./spnav-0.9-py2.7.egg
    ./uncertainties-2.4.4-py2.7.egg
    ./Jinja2-2.7.2-py2.7.egg
    ./Pygments-1.6-py2.7.egg
    ./pycollada-0.4-py2.7.egg
    ./readline-6.2.4.1-py2.7-macosx-10.9-x86_64.egg
    /usr/local/env/chroma_env/src/chroma
    import sys; new=sys.path[sys.__plen:]; del sys.path[sys.__plen:]; p=getattr(sys,'__egginsert',0); sys.path[p:p]=new; sys.__egginsert = p+len(new)
    delta:site-packages blyth$ 


Probably the **install** command created the egg, but the recommended way is **develop** which plans the path 
in the `easy-install.pth`:: 

    delta:site-packages blyth$ find /usr/local/env/chroma_env/src/chroma -name geometry_types.h
    /usr/local/env/chroma_env/src/chroma/chroma/cuda/geometry_types.h

Removing the header from the working copy reproduces the error, so the correct header now
being used. There was some flakiness in this, supect an egg cache may be coming into play.


chroma-deps pycuda build failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://wiki.tiker.net/PyCuda/Installation/Mac

While doing::

   pip install -b /usr/local/env/chroma_env/build/build_pycuda pycuda

#. linker errors from missing dir /Developer/NVIDIA/CUDA-5.5/lib64, avoid with ~/.aksetup-defaults.py change
  
#. linker errors from missing lib "-lcuda", avoid by changing libdir to /usr/local/cuda/lib  

#. pyublas complains: "Setuptools conflict detected", but it proceeds anyhow

   * http://wiki.tiker.net/DistributeVsSetuptools 

#. pip/shrinkwrap/cmake builds dump lots of whitespace, possibly colorful clang warnings, seems 
   worse with iTerm2.app term setup rather than with Terminal.app 

#. cmake-2.8.11 build issue with cmlibarchive

   * https://github.com/Kitware/CMake/tree/master/Utilities/cmlibarchive

   * try grabbing that externally via macports::

        sudo port -v install cmake  # 

   * macports cmake is ignored, the http://www.cmake.org/files/v2.8/cmake-2.8.11.tar.gz being attempted to be built again

   * duplicate shrinkwrap repo on delta using shrinkwrap-dupe 
     and add new geant4-4.9.5.post2.tar.gz which is same as post1 but 
     with shrinkwrap_requires=["cmake"] commented and with the new version name

     * with this succeed to complete geant4 build

   * subsequent root build fails with freetype header missing 


root 5.34.11 freetype issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    In file included from /usr/local/env/chroma_env/src/root-v5.34.11/graf2d/x11ttf/src/TGX11TTF.cxx:28:
    In file included from include/TGX11TTF.h:34:
    In file included from include/TTF.h:30:
    /usr/X11R6/include/ft2build.h:56:10: fatal error: 'freetype/config/ftheader.h' file not found
    #include <freetype/config/ftheader.h>

Looks like a missing 2::

    delta:lib blyth$ freetype-config --cflags
    -I/opt/local/include/freetype2
    delta:lib blyth$ ll /opt/local/include/freetype2/config/ftheader.h 
    -rw-r--r--  1 root  admin  25587 Dec 22 17:32 /opt/local/include/freetype2/config/ftheader.h

The /usr/X11R6/include/ft2build.h header with the wrong path::

     55   /* `<prefix>/include/freetype2' must be in your current inclusion path */
     56 #include <freetype/config/ftheader.h>

::

    freetype-config --cflags
    -I/opt/local/include/freetype2

Perhaps an XQuartz ROOT incompatibility, try kludge::

   cd /opt/local/include/freetype2 && sudo ln -s . freetype

Makes the header appear at locations expected by both ROOT and Xquartz::

    (chroma_env)delta:freetype2 blyth$ ls -l /opt/local/include/freetype2/config/ftheader.h /opt/local/include/freetype2/freetype/config/ftheader.h
    -rw-r--r--  1 root  admin  25587 Dec 22 17:32 /opt/local/include/freetype2/config/ftheader.h
    -rw-r--r--  1 root  admin  25587 Dec 22 17:32 /opt/local/include/freetype2/freetype/config/ftheader.h




* http://trac.macports.org/ticket/41746
* http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=17190

* https://xquartz.macosforge.org/trac/wiki/Releases 

  * promulgates X11 2.7.5 - 2013.11.10 - First release supported on Mavericks
  * also mentions use of macports xorg-server or xorg-server-devel
  * https://trac.macports.org/browser/trunk/dports/x11/xorg-server-devel
  * https://trac.macports.org/browser/trunk/dports/x11/xorg-server


* http://root.cern.ch/drupal/content/root-version-v5-34-00-patch-release-notes

  * claims v5-34-12 (Nov 19, 2013) completes Mavericks support



rerun root
~~~~~~~~~~~~

A rerun skips root unless::

   rm -rf build/build_root

Do this with *chroma-deps-rebuild root*


g4py_chroma
~~~~~~~~~~~~~~

Claims to complete OK but looks like python version mixing between
system and macports python::

        Checking for system type ... macosx
        Checking for prefix ... /usr/local/env/chroma_env
        Checking for lib dir ... /usr/local/env/chroma_env/lib/python2.7/site-packages
        Checking for G4 include dir ... /usr/local/env/chroma_env/include/Geant4
        Checking for G4 lib dir ... /usr/local/env/chroma_env/lib/
        Checking for G4 libs are shared libraray ... ok
        Checking for Python include dir (pyconfig.h) ... /usr/include/python2.7
        Checking for Python lib dir ... /usr/local/env/chroma_env/lib/python2.7/config
        Checking for Boost include dir (boost/python.hpp) ... /usr/local/env/chroma_env/include
        Checking for Boost version ... ok
        Checking for Boost lib dir ... /usr/local/env/chroma_env/lib
        Checking for Boost Python lib name ... libboost_python.dylib
        Checking for OpenGL support ...no
        Checking for GL2PS support ...no
        Checking for physics list support ...yes
        Checking for GDML support ...no


Also some funny warnings::

        Building a module G4global.so ...
        ld: warning: directory not found for option '-L-l'
        ... intall G4global.so into /usr/local/env/chroma_env/lib/python2.7/site-packages/Geant4

pyzmq
~~~~~

Claims to complete OK despite issue with underlying ZMQ which gave fatality::

        build/temp.macosx-10.9-x86_64-2.7/scratch/vers.c:4:10: fatal error: 'zmq.h' file not found
        #include "zmq.h"
                 ^
        1 error generated.
    
        error: command '/usr/bin/clang' failed with exit status 1
    
        Warning: Failed to build or run libzmq detection test.
    
        If you expected pyzmq to link against an installed libzmq, please check to make sure:
    
            * You have a C compiler installed
            * A development version of Python is installed (including headers)
            * A development version of ZMQ >= 2.1.4 is installed (including headers)
            * If ZMQ is not in a default location, supply the argument --zmq=<path>
            * If you did recently install ZMQ to a default location,
              try rebuilding the ld cache with `sudo ldconfig`
              or specify zmq's location with `--zmq=/usr/local`
    
        If you expected to get a binary install (egg), we have those for
        current Pythons on OS X and Windows. These can be installed with
        easy_install, but PIP DOES NOT SUPPORT EGGS.
    
        You can skip all this detection/waiting nonsense if you know
        you want pyzmq to bundle libzmq as an extension by passing:
    
            `--zmq=bundled`
    
        I will now try to build libzmq as a Python extension
        unless you interrupt me (^C) in the next 10 seconds...
    
        ************************************************
        Using bundled libzmq
        already have bundled/zeromq
        attempting ./configure to generate platform.hpp
        Warning: failed to configure libzmq:
        /bin/sh: ./configure: No such file or directory


* https://github.com/zeromq/pyzmq/wiki/Building-and-Installing-PyZMQ

ZMQ
~~~~

As now interested in ZMQ from C++ need the zmq.h header, so install
that with *zeromq-;zeromq-make*

* https://github.com/zeromq/pyzmq/issues/500



markupsafe 
~~~~~~~~~~~

pip installing it separately gets around the SandboxViolation when installed
as a dependency of chroma::

    Installed /usr/local/env/chroma_env/lib/python2.7/site-packages/Pygments-1.6-py2.7.egg
    Searching for markupsafe
    Reading https://pypi.python.org/simple/markupsafe/
    Best match: MarkupSafe 0.18
    Downloading https://pypi.python.org/packages/source/M/MarkupSafe/MarkupSafe-0.18.tar.gz#md5=f8d252fd05371e51dec2fe9a36890687
    Processing MarkupSafe-0.18.tar.gz
    Writing /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-IYqptm/MarkupSafe-0.18/setup.cfg
    Running MarkupSafe-0.18/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-IYqptm/MarkupSafe-0.18/egg-dist-tmp-DMeghv
    error: Setup script exited with error: SandboxViolation: os.open('/var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/tmpKdJUob/SDCV5z', 2818, 384) {}

    The package setup script has attempted to modify files on your system
    that are not within the EasyInstall build area, and has been aborted.

    This package cannot be safely installed by EasyInstall, and may not
    support alternate installation locations even if you run its setup
    script by hand.  Please inform the package's author and the EasyInstall
    maintainers to find out if a fix or workaround is available.



pip logs
~~~~~~~~~

Are overritten at each invokation::

   mv /Users/blyth/.pip/pip.log ~/chroma_deps.log


root-5.34.14
~~~~~~~~~~~~~~~

Bump root to get Mavericks fixes, by adding root-5.34.14.tar.gz 
to chroma_pkgs with the only change being the version at the 
below three points.::

    chroma_env)delta:root blyth$ diff -r  root-5.34.11 root-5.34.14
    diff -r root-5.34.11/PKG-INFO root-5.34.14/PKG-INFO
    3c3
    < Version: 5.34.11
    ---
    > Version: 5.34.14
    diff -r root-5.34.11/root.egg-info/PKG-INFO root-5.34.14/root.egg-info/PKG-INFO
    3c3
    < Version: 5.34.11
    ---
    > Version: 5.34.14
    diff -r root-5.34.11/setup.py root-5.34.14/setup.py
    10c10
    < version = '5.34.11'
    ---
    > version = '5.34.14'


::

    (chroma_env)delta:chroma_env blyth$ chroma-deps-rebuild root -U
    === chroma-deps-env : writing /Users/blyth/.aksetup-defaults.py
    Downloading/unpacking root from http://localhost/chroma_pkgs/root/root-5.34.14.tar.gz
      Downloading root-5.34.14.tar.gz
      Running setup.py egg_info for package root
        
    Installing collected packages: root
      Found existing installation: root 5.34.11
        Uninstalling root:
          Successfully uninstalled root
      Running setup.py install for root
        Downloading ftp://root.cern.ch/root/root_v5.34.14.source.tar.gz


TTF : header name clash between freetype and ftgl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

     6860 
     6861     In file included from /usr/local/env/chroma_env/src/root-v5.34.14/graf2d/graf/src/TMathText.cxx:15:
     6862 
     6863     include/TTF.h:51:4: error: unknown type name 'FT_Glyph'; did you mean 'FTGlyph'?
     6864 
     6865        FT_Glyph   fImage; // glyph image
     6866 
     6867        ^~~~~~~~
     6868 
     6869        FTGlyph
     6870 
     6871     include/ftglyph.h:25:19: note: 'FTGlyph' declared here
     6872 
     6873     class FTGL_EXPORT FTGlyph
     6874 

* http://trac.macports.org/ticket/41572
* https://github.com/root-mirror/root/commit/446a11828dcf577efd15d9057703c5bd099dd148
* https://sft.its.cern.ch/jira/browse/ROOT-5773
* http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=17485



rebuild pycuda with opengl capability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trying to run PyCUDA OpenGL interop example

* http://andreask.cs.illinois.edu/PyCuda/Examples/GlInterop

Gives::

     (chroma_env)delta:pycuda_pyopengl_interop blyth$ python interop.py 
     ...
     import pycuda.gl as cuda_gl
     File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/gl/__init__.py", line 4, in <module>
     raise ImportError("PyCUDA was compiled without GL extension support")


#. add flag to chroma-pycuda-aksetup- and regenerate::

    (chroma_env)delta:pycuda_pyopengl_interop blyth$ chroma-pycuda-aksetup
    writing /Users/blyth/.aksetup-defaults.py

::

   pip uninstall pycuda
   chroma-deps-rebuild pycuda


Chroma Install
----------------

* http://chroma.bitbucket.org/install/overview.html#common-install

Suggestion is::

    cd $VIRTUAL_ENV/src
    hg clone https://bitbucket.org/chroma/chroma
    cd chroma
    python setup.py develop

Chroma Rebuild
---------------



Chroma Install Rerun from my bitbucket fork
-----------------------------------------------

::

    cd $VIRTUAL_ENV/src
    hg clone ssh://hg@bitbucket.org/scb-/chroma
    cd chroma
    # python setup.py develop   ## uninstall first ?

::

    (chroma_env)delta:~ blyth$ chroma-
    (chroma_env)delta:~ blyth$ chroma-cd src/chroma
    (chroma_env)delta:chroma blyth$ pwd
    /usr/local/env/chroma_env/src/chroma

    (chroma_env)delta:chroma blyth$ python setup.py --help   

      ## no uninstall command, the "develop" command just plants an egg-link anyhow

    (chroma_env)delta:chroma blyth$ cat ../../lib/python2.7/site-packages/Chroma.egg-link 
    /usr/local/env/chroma_env/src/chroma

    .(chroma_env)delta:chroma blyth$ cat ../../lib/python2.7/site-packages/easy-install.pth 
    import sys; sys.__plen = len(sys.path)
    /usr/local/env/chroma_env/src/chroma
    ./unittest2-0.5.1-py2.7.egg
    ./Sphinx-1.2-py2.7.egg
    ./spnav-0.9-py2.7.egg
    ./uncertainties-2.4.4-py2.7.egg
    ./Jinja2-2.7.2-py2.7.egg
    ./Pygments-1.6-py2.7.egg
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages
    ./pycollada-0.4-py2.7.egg
    ./readline-6.2.4.1-py2.7-macosx-10.9-x86_64.egg
    import sys; new=sys.path[sys.__plen:]; del sys.path[sys.__plen:]; p=getattr(sys,'__egginsert',0); sys.path[p:p]=new; sys.__egginsert = p+len(new)
    (chroma_env)delta:chroma blyth$ 


Plow ahead with the `develop` ontop of existing install::

    (chroma_env)delta:chroma blyth$ python setup.py develop
    running develop
    running egg_info
    creating Chroma.egg-info
    writing requirements to Chroma.egg-info/requires.txt
    writing Chroma.egg-info/PKG-INFO
    ... 
    Using /usr/local/env/chroma_env/lib/python2.7/site-packages
    Finished processing dependencies for Chroma==0.5


Moved retainable chroma mods into bitbucket fork
-------------------------------------------------

* only significant mods canned were to camera.py 
  as the over complicated nature of that makes it 
  not a good place for development without a major re-write
   


EOU
}
chroma-dir(){ 
   case $NODE_TAG in 
      D) echo $(local-base)/env/chroma_env ;;
      *) echo $(local-base)/env/chroma ;;
   esac
}

chroma-sdir(){
   case $NODE_TAG in 
      D) echo $(chroma-dir)/src/chroma/chroma ;;
      *) echo $(chroma-dir)/chroma ;;
   esac
}

chroma-scd(){  cd $(chroma-sdir)/$1 ; }
chroma-env(){      
    elocal-  
    local dir=$(chroma-dir)
    [ -f "$dir/bin/activate" ] && source $dir/bin/activate 

    cuda-  # hmm dirty, perhaps do via shrinkwrap $VIRTUAL_ENV/env.d ??
}
chroma-cd(){  cd $(chroma-dir)/$1; }
chroma-mate(){ mate $(chroma-dir) ; }
chroma-get(){
   [ "$NODE_TAG" == "D" ] && echo $msg NOT USED ON NODE $NODE_TAG SEE chroma-build && return 1
   local dir=$(dirname $(chroma-dir)) &&  mkdir -p $dir && cd $dir
   hg clone https://bitbucket.org/chroma/chroma
}


chroma-prepare(){
    chroma-virtualenv
    chroma-versions
    chroma-deps
}

chroma-versions(){
    which python
    python -V
    which virtualenv
    virtualenv --version
}

chroma-virtualenv(){
    local msg="=== $FUNCNAME :"
    local dir=$(chroma-dir)
    [ -d $dir ] && echo $msg chroma virtualenv dir exists already $dir && return 0

    # want access to macports py27 modules like numpy/pygame/matplotlib/... 
    # so use the somewhat dirty --system-site-package option

    virtualenv --system-site-package  $(chroma-dir)    
}

chroma-pycuda-aksetup(){  
   local out=~/.aksetup-defaults.py
   echo $msg writing $out
   $FUNCNAME- > $out
 }
chroma-pycuda-aksetup-(){  
   cuda-
   cat << EOS
# $FUNCNAME
import os
virtual_env = os.environ['VIRTUAL_ENV']
cuda_root = '$(cuda-dir)'
cuda_lib_dir = [os.path.join(cuda_root,'lib')]
cuda_lib_dir = ['/usr/local/cuda/lib']
BOOST_INC_DIR = [os.path.join(virtual_env, 'include')]
BOOST_LIB_DIR = [os.path.join(virtual_env, 'lib')]
BOOST_PYTHON_LIBNAME = ['boost_python']

# guess based on 
#   http://wiki.tiker.net/PyCuda/Installation/Mac
#   https://github.com/inducer/pycuda/blob/master/setup.py
#
CUDADRV_LIB_DIR = cuda_lib_dir
CUDART_LIB_DIR = cuda_lib_dir
CURAND_LIB_DIR = cuda_lib_dir

# for pycuda OpenGL interop
CUDA_ENABLE_GL = True


EOS
}

chroma-indexurl(){
   #echo http://mtrr.org/chroma_pkgs/
   #echo http://localhost/chroma_pkgs/
   echo http://localhost/chroma_pkgs/
   #echo file:///usr/local/env/chroma_pkgs/    NOPE: pip needs a webserver to provide an index.html
}

chroma-deps-env(){
   local msg="=== $FUNCNAME :"
   [ -z "$VIRTUAL_ENV" ] && echo $msg ERROR need to be in the virtualenv to proceed && return 1
   cuda-    # PATH setup for CUDA, expect ignored however these setting coming from aksetup ?
   chroma-pycuda-aksetup
   export PIP_EXTRA_INDEX_URL=$(chroma-indexurl)

   xercesc-
   xercesc-geant4-export
   env | grep XERCESC
}

chroma-deps(){
   chroma-  # activate the virtualenv

   chroma-deps-env
   cd $VIRTUAL_ENV  
   pip install chroma_deps
}



chroma-deps-rebuild-geant4(){ 

    rm -rf $(chroma-dir)/build/build_geant4
    rm -rf lib/python2.7/site-packages/geant4-4.9.5.post2-py2.7.egg-info
    rm -rf $(chroma-dir)/src/geant4.9.5.p01   
    rm -rf $(chroma-dir)/include/Geant4

    chroma-deps-rebuild geant4 -U -v --pre $*

}
chroma-deps-rebuild(){
   local msg="=== $FUNCNAME :"
   local name=${1:-root}
   shift
   local args=$*
   local builddir=$VIRTUAL_ENV/build/build_$name
   [ -d "$builddir" ] && echo $msg builddir for $name exists already : $builddir : delete this and rerun to proceed && return 1

   chroma-deps-env
   cd $VIRTUAL_ENV  
   local cmd="pip install -b $builddir --index-url $(chroma-indexurl) $args $name"
   echo $msg $cmd 
   eval $cmd
}





chroma-kludge-root(){
   chroma-


   #cd $VIRTUAL_ENV/src/root-v5.34.14.patch01
   #./configure --enable-minuit2 --enable-roofit --with-python-libdir=$VIRTUAL_ENV/lib/python2.7/config
   # nope getting same issue

   local version=5.34.14
   cd $VIRTUAL_ENV/src/root-v$version
   # workaround is to use builtin freetype
   ./configure --enable-builtin-freetype  --enable-minuit2 --enable-roofit --with-python-libdir=$VIRTUAL_ENV/lib/python2.7/config
   make -j8 

   echo "source $VIRTUAL_ENV/src/root-v$version/bin/thisroot.sh" > $VIRTUAL_ENV/env.d/root.sh

   chroma-  # pick up new env
   env | grep ROOTSYS

}


chroma-build(){
   local msg="=== $FUNCNAME :"
   chroma-
   [ -z "$VIRTUAL_ENV" ] && echo $msg ERROR need to be in the virtualenv to proceed && return 1


   pip install markupsafe # gives SandboxViolation when installing as dependency of chroma/Sphinx/Pygments 

   cd $VIRTUAL_ENV/src
   [ ! -d chroma ] && hg clone https://bitbucket.org/chroma/chroma
   cd chroma
   python setup.py develop


}


chroma-rebuild(){
   cd $VIRTUAL_ENV/src/chroma
   hg paths
   hg status
   python setup.py develop
}




### TRYING TO BUILD GEANT4 EXAMPLES AGAINST THE CHROMA GEANT4

chroma-geant4-name(){ echo geant4.9.5.p01 ;}
chroma-geant4-sdir(){ echo $(chroma-dir)/src/$(chroma-geant4-name) ; }

chroma-geant4-builddir(){ echo $(chroma-dir)/src/geant4.9.5.p01-build ; }
chroma-geant4-builddir-cd(){ cd $(chroma-geant4-builddir) ; }

chroma-geant4-dir(){  echo $(chroma-dir)/lib/Geant4-9.5.1 ; }
chroma-geant4-dir-check(){  
   [ -f "$(chroma-geant4-dir)/Geant4Config.cmake" ] && echo $FUNCNAME OK || echo $FUNCNAME ERROR 
}
chroma-geant4-cd(){ cd  $(chroma-geant4-dir) ; }
chroma-geant4-scd(){ cd $(chroma-geant4-sdir) ; }

chroma-geant4-export(){
   export CHROMA_GEANT4_SDIR=$(chroma-geant4-sdir)
   export GEANT4_HOME=$(chroma-geant4-sdir) # needed by several CMakeLists.txt   see geant4sys-
   #env | grep CHROMA_GEANT4
}

chroma-geant4-findroot(){ echo $(chroma-geant4-sdir)/cmake/Modules/FindROOT.cmake ; }
chroma-geant4-findroot-vi(){ vi $(chroma-geant4-findroot) ; }

chroma-clhep-prefix(){
   echo $VIRTUAL_ENV/src/$(chroma-geant4-name)/source/externals/clhep 
}

chroma-g4-bdir(){      echo $(chroma-dir)/src/$(chroma-geant4-name) ; }
chroma-g4-incdir(){    echo $(chroma-dir)/include/Geant4 ; }
chroma-g4-libdir(){    echo $(chroma-dir)/src/$(chroma-geant4-name)-build/outputs/library/Darwin-UNSUPPORTED ; }

chroma-clhep-incdir(){ echo $(chroma-dir)/include/Geant4/CLHEP ; }
chroma-clhep-libdir(){ echo $(chroma-g4-libdir) ; }   ## incorporated with G4? /usr/local/env/chroma_env/lib/libG4clhep.dylib 

chroma-xercesc-incdir(){ xercesc- ; echo $(xercesc-include-dir) ; }
chroma-xercesc-libdir(){ xercesc- ; echo $(dirname $(xercesc-library)) ; }
  

chroma-root-prefix(){ echo $(chroma-dir)/src/root-v5.34.11 ; }
chroma-root-incdir(){ echo $(chroma-root-prefix)/include ; }
chroma-root-libdir(){ echo $(chroma-root-prefix)/lib ; }


 
