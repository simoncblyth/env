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

Claims to complete OK despite this fatality::

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




EOU
}
chroma-dir(){ echo $(local-base)/env/chroma_env ; }
chroma-env(){      
    elocal-  
    local dir=$(chroma-dir)
    [ -d $dir ] && source $dir/bin/activate 

    cuda-  # hmm dirty, perhaps do via shrinkwrap $VIRTUAL_ENV/env.d ??
}
chroma-cd(){  cd $(chroma-dir); }
chroma-mate(){ mate $(chroma-dir) ; }
chroma-get(){
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

EOS
}

chroma-pkgs-url(){
   #echo http://mtrr.org/chroma_pkgs/
   echo http://localhost/chroma_pkgs/
}

chroma-deps-env(){
   local msg="=== $FUNCNAME :"
   [ -z "$VIRTUAL_ENV" ] && echo $msg ERROR need to be in the virtualenv to proceed && return 1
   cuda-    # PATH setup for CUDA, expect ignored however these setting coming from aksetup ?
   chroma-pycuda-aksetup
   export PIP_EXTRA_INDEX_URL=$(chroma-pkgs-url)
}

chroma-deps(){
   chroma-  # activate the virtualenv

   chroma-deps-env
   cd $VIRTUAL_ENV  
   pip install chroma_deps
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
   pip install -b $builddir $args $name
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


