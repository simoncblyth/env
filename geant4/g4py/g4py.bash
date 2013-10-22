# === func-gen- : geant4/g4py/g4py fgp geant4/g4py/g4py.bash fgn g4py fgh geant4/g4py
g4py-src(){      echo geant4/g4py/g4py.bash ; }
g4py-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4py-src)} ; }
g4py-vi(){       vi $(g4py-source) $* ; }
g4py-usage(){ cat << EOU

Geant4Py
=========

Executive Summary
------------------

After considerable installation woes documented below, it turns out 
that g4py misses the boost_python glue to expose the `G4VSolid::GetPolygons` 
that is needed to extract quads/tris from Geant4 geometry.  Possibly that 
is a simple omission, or exposing that might require exposing other
classes. 

Once reaching this impasse, I reverted to using standard Geant4 in memory 
geometry tree as the starting point to creating a COLLADA 
exporter based upon the GDML exporter.

For development convenience the geometry is loaded from a
GDML file.

Sources/Examples
------------------

* https://github.com/nepahwin/Geant4.9.6/blob/master/environments/g4py/examples/gdml/write_gdml.py
* http://geant4.cern.ch/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/apas09.html
* http://bugzilla-geant4.kek.jp/show_bug.cgi?id=1317#c4
* https://bitbucket.org/seibert/g4py/commits/all

g4py configure has issues
----------------------------

The LCG configure is plumping for system python include dir even when 
give the argument pointing elsewhere. Prevent this with "/no/way/jose".

external/build/LCG/geant4.9.2.p01/environments/g4py/configure::

    270 echo $ac_n "Checking for Python include dir (pyconfig.h) ... $ac_c"
    271 # check version
    272 set python python2.5 python2.4 python2.3 python2.2
    273 for aincdir in $*
    274 do
    275   if [ -d /no/way/jose/usr/include/$aincdir ]; then
    276     python_incdir=/usr/include/$aincdir
    277     break
    278   fi
    279 done

Install on N 
--------------

Build needs geant4 global libs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Missing libG4persistency::

    Building a module G4global.so ...
    /usr/bin/ld: cannot find -lG4persistency
    collect2: ld returned 1 exit status
    make[2]: *** [G4global.so] Error 1

Make it using the global target from source/persistency and manual install::

    [blyth@belle7 persistency]$ make CLHEP_BASE_DIR=$DYB/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$DYB/external/XercesC/2.8.0/i686-slc5-gcc41-dbg global
    Nothing to be done for libG4persistency in mctruth/.
    Nothing to be done for libG4persistency in ascii/.
    Nothing to be done for libG4persistency in gdml/.
    Creating global shared library ../../lib/Linux-g++/libG4persistency.so ...

    [blyth@belle7 persistency]$ cp ../../lib/Linux-g++/libG4persistency.so $DYB/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/

After fix lack of persistency lib get::

    /usr/bin/ld: cannot find -lG4parmodels
    [blyth@belle7 source]$ find . -name GNUmakefile  -exec grep -H parmodels {} \;
    ./parameterisations/GNUmakefile:name := G4parmodels

    [blyth@belle7 parameterisations]$ make CLHEP_BASE_DIR=$DYB/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1  global
    Nothing to be done for libG4parmodels in gflash/.
    Creating global shared library ../../lib/Linux-g++/libG4parmodels.so ...
    [blyth@belle7 parameterisations]$ cp ../../lib/Linux-g++/libG4parmodels.so $DYB/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/

Subsequently miss libG4processes, need all the global libs it seems::

    [blyth@belle7 source]$ make CLHEP_BASE_DIR=$DYB/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1  global
    [blyth@belle7 source]$ cp ../lib/Linux-g++/libG4persistency.so ../lib/Linux-g++/libG4parmodels.so ../lib/Linux-g++/libG4global.so ../lib/Linux-g++/libG4geometry.so ../lib/Linux-g++/libG4particles.so ../lib/Linux-g++/libG4processes.so ../lib/Linux-g++/libG4physicslists.so ../lib/Linux-g++/libG4interfaces.so $DYB/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/

Subsequently miss digits_hits, pilot error::
 
     [blyth@belle7 source]$ cp ../lib/Linux-g++/libG4digits_hits.so $DYB/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/
 

Wrong boost_python
~~~~~~~~~~~~~~~~~~~~~

After building global g4 libs succeed to build and install. But somehow the system, libboost_python is stepping in::

    [blyth@belle7 ~]$ PYTHONPATH=$(g4py-dir)/lib python -c "import Geant4 "
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/environments/g4py/lib/Geant4/__init__.py", line 19, in <module>
        from G4interface import *
    ImportError: /usr/lib/libboost_python.so.2: undefined symbol: PyUnicodeUCS4_FromEncodedObject
    [blyth@belle7 ~]$ 


::

    [blyth@belle7 source]$ g4py-configure
    Checking for system type ... linux
    Checking for prefix ... /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/environments/g4py
    Checking for lib dir ... /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/environments/g4py/lib
    Checking for G4 include dir ... /data1/env/local/dyb/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/include
    Checking for G4 lib dir ... /data1/env/local/dyb/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib
    Checking for CLHEP include dir ... /data1/env/local/dyb/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg/include
    Checking for CLHEP lib dir ... /data1/env/local/dyb/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg/lib
    Checking for CLHEP lib name ... libCLHEP-2.0.4.2.so
    Checking for Python include dir (pyconfig.h) ... /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/include/python2.7
    Checking for Python lib dir ... /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/lib
    Checking for Boost include dir (boost/python.hpp) ... /data1/env/local/dyb/external/Boost/1.38.0_python2.7/i686-slc5-gcc41-dbg/include/boost-1_38
    Checking for Boost version ... ok
    Checking for Boost lib dir ... /data1/env/local/dyb/external/Boost/1.38.0_python2.7/i686-slc5-gcc41-dbg/lib
    Checking for Boost Python lib name ... libboost_python-gcc41-mt.so
    Checking for OpenGL support ...yes
    Checking for physics list support ...yes
    Checking for GDML support ...yes
    Checking for Xerces-C include dir ...yes
    Checking for Xerces-C lib dir ...yes
    Writing config.gmk ... done
    Writing config.status ... done

    Enabled support for openglx physicslist gdml.

    To build Geant4Py type:

      make
      make install

Maybe rpath problem::

    [blyth@belle7 g4py]$ pwd
    /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/environments/g4py/lib/g4py
    [blyth@belle7 g4py]$ ldd *.so | grep boost
        libboost_python.so.2 => /usr/lib/libboost_python.so.2 (0x00110000)
        libboost_python.so.2 => /usr/lib/libboost_python.so.2 (0x0083d000)
        libboost_python.so.2 => /usr/lib/libboost_python.so.2 (0x00745000)
        ...

Up verbosity, shows the link is using libboost_python.so despite the config telling it otherwise::

    [blyth@belle7 g4py]$ make CPPVERBOSE=1

    g++ -m32 -Wl,-rpath,/data1/env/local/dyb/external/Boost/1.38.0_python2.7/i686-slc5-gcc41-dbg/lib:/data1/env/local/dyb/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib:/data1/env/local/dyb/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg/lib: 
         -Wl,-soname,G4track.so -shared -o 
             G4track.so 
            pyG4Step.o  pyG4StepPoint.o  pyG4StepStatus.o  pyG4Track.o  pyG4TrackStatus.o  pymodG4track.o  
            -L/data1/env/local/dyb/external/Boost/1.38.0_python2.7/i686-slc5-gcc41-dbg/lib -lboost_python 
            -L/data1/env/local/dyb/external/XercesC/2.8.0/i686-slc5-gcc41-dbg/lib -lxerces-c 
            -L/data1/env/local/dyb/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib -lG4persistency -lG4readout -lG4run -lG4event -lG4tracking 
                         -lG4parmodels -lG4processes -lG4digits_hits -lG4track -lG4particles -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms 
                         -lG4interfaces -lG4global -lG4physicslists  -lG4FR -lG4visHepRep -lG4RayTracer -lG4VRML -lG4Tree -lG4OpenGL -lG4vis_management -lG4modeling
            -L/data1/env/local/dyb/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg/lib -lCLHEP-2.0.4.2


Another config buglet $DYB/external/build/LCG/geant4.9.2.p01/environments/g4py/config/module.gmk::

     35 #LOPT  += -lboost_python
     36 ifdef Q_BOOST_PYTHON_LIB
     37    LOPT += -l$(Q_BOOST_PYTHON_LIB)
     38 endif

Now onto a boost python problem rather than a config one::

    [blyth@belle7 env]$ g4py-test
    /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/environments/g4py/lib/Geant4/__init__.py", line 21, in <module>
        from G4global import *
    AttributeError: 'Boost.Python.StaticProperty' object attribute '__doc__' is read-only

    [blyth@cms01 cmt]$ g4py-test
    pypath /data/env/local/dyb/trunk/external/Python/2.7/i686-slc4-gcc34-dbg/bin/python is OK
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/environments/g4py/lib/Geant4/__init__.py", line 21, in <module>
        from G4global import *
    AttributeError: 'Boost.Python.StaticProperty' object attribute '__doc__' is read-only


* https://bugs.launchpad.net/ubuntu/+source/boost1.38/+bug/457688
* https://svn.boost.org/trac/boost/changeset/53731/sandbox-branches/bhy/py3k


On C
~~~~~~

::

    [blyth@cms01 g4py]$ ./conftest
    ./conftest: error while loading shared libraries: libboost_python.so.1.54.0: cannot open shared object file: No such file or directory
    [blyth@cms01 g4py]$ 
    [blyth@cms01 g4py]$ LD_LIBRARY_PATH=/data/env/local/env/boost/boost_1_54_0.local/lib:$LD_LIBRARY_PATH ./conftest


Another g4py configure kludge::

    373 echo g++ -o conftest -I$boost_incdir -I$python_incdir -L$boost_libdir -l$boost_python_lib -L$python_libdir -lpython2.7 conftest.cc 
    374 g++ -o conftest -I$boost_incdir -I$python_incdir -L$boost_libdir -l$boost_python_lib -L$python_libdir -lpython2.7 conftest.cc
    375 g++ -o conftest -I$boost_incdir -I$python_incdir -L$boost_libdir -l$boost_python_lib -L$python_libdir -lpython2.7 conftest.cc > /dev/null 2>&1
    376 LD_LIBRARY_PATH=$boost_libdir:$LD_LIBRARY_PATH ./conftest
    377 q_boost_version=$?



Finally
~~~~~~~~~

After building boost_python from latest boost 1_54_0 succeed to import::

    [blyth@belle7 g4py]$ PYTHONPATH=$(g4py-libdir):$PYTHONPATH ipython
    Python 2.7 (r27:82500, Feb 16 2011, 11:40:18) 
    Type "copyright", "credits" or "license" for more information.

    IPython 0.9.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object'. ?object also works, ?? prints more.

    In [1]: import Geant4
    /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/environments/g4py/lib/Geant4/__init__.py:29: RuntimeWarning: to-Python converter for std::vector<G4Element*, std::allocator<G4Element*> > already registered; second conversion method ignored.
      from G4materials import *

    *************************************************************
     Geant4 version Name: geant4-09-02-patch-01    (13-March-2009)
                          Copyright : Geant4 Collaboration
                          Reference : NIM A 506 (2003), 250-303
                                WWW : http://cern.ch/geant4
    *************************************************************

    Visualization Manager instantiating...

    In [2]: 


    [blyth@cms01 g4py]$ g4py-test
    pypath /data/env/local/dyb/trunk/external/Python/2.7/i686-slc4-gcc34-dbg/bin/python is OK
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/environments/g4py/lib/Geant4/__init__.py:29: RuntimeWarning: to-Python converter for std::vector<G4Element*, std::allocator<G4Element*> > already registered; second conversion method ignored.
      from G4materials import *

    *************************************************************
     Geant4 version Name: geant4-09-02-patch-01    (13-March-2009)
                          Copyright : Geant4 Collaboration
                          Reference : NIM A 506 (2003), 250-303
                                WWW : http://cern.ch/geant4
    *************************************************************

    Visualization Manager instantiating...



But still issue on C
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 gdml]$ ./g4gdml.py 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/environments/g4py/lib/Geant4/__init__.py:29: RuntimeWarning: to-Python converter for std::vector<G4Element*, std::allocator<G4Element*> > already registered; second conversion method ignored.
      from G4materials import *

    *************************************************************
     Geant4 version Name: geant4-09-02-patch-01    (13-March-2009)
                          Copyright : Geant4 Collaboration
                          Reference : NIM A 506 (2003), 250-303
                                WWW : http://cern.ch/geant4
    *************************************************************

    Visualization Manager instantiating...
    Traceback (most recent call last):
      File "./g4gdml.py", line 63, in <module>
        prs = Geant4.G4GDMLParser()
    AttributeError: 'module' object has no attribute 'G4GDMLParser'
    [blyth@cms01 gdml]$ 


Argh, GDML needs to be built. Specifically need libG4gdml.so and libG4persistency.so



FUNCTIONS
-----------

g4py-g4global
              builds the geant4 global libs, these are needed for g4py 



EOU
}
g4py-env(){     
    elocal- 
    nuwa-
    boost- 
}
g4py-dir(){ echo $(nuwa-g4-bdir)/environments/g4py ; }
g4py-cd(){  cd $(g4py-dir); }
g4py-mate(){ mate $(g4py-dir) ; }
g4py-get(){
   local dir=$(dirname $(g4py-dir)) &&  mkdir -p $dir && cd $dir
}

g4py-prefix(){ echo $(g4py-dir) ; }
g4py-libdir(){ echo $(g4py-prefix)/lib ; }


g4py-configure(){
   cd $(g4py-dir)
   ./configure  linux \
                  --prefix=$(g4py-prefix) \
                  --libdir=$(g4py-libdir) \
                  --with-g4-incdir=$(nuwa-g4-incdir) \
                   --with-g4-libdir=$(nuwa-g4-libdir) \
                --with-clhep-incdir=$(nuwa-clhep-incdir) \
                --with-clhep-libdir=$(nuwa-clhep-libdir) \
                   --with-clhep-lib=$(nuwa-clhep-lib) \
                --with-boost-incdir=$(boost-incdir) \
                --with-boost-libdir=$(boost-libdir) \
                --with-xercesc-incdir=$(nuwa-xercesc-incdir) \
                --with-xercesc-libdir=$(nuwa-xercesc-libdir) \
                --with-python-incdir=$(nuwa-python-incdir) \
                --with-python-libdir=$(nuwa-python-libdir) \
                --with-boost-python-lib=$(boost-python-lib) \
                --with-extra-dir=/dev/null

}


g4py-pycheck(){
   local pypath=$(which python)  
   case $pypath in 
     $DYB*) echo pypath $pypath is OK && return 0  ;;
         *) echo try again with the python that are building against in the path not $pypath && return 1;;
   esac
}

g4py-make(){
   cd $(g4py-dir)
   g4py-pycheck && make CPPVERBOSE=1
}

g4py-install(){
   cd $(g4py-dir)
   g4py-pycheck && make install CPPVERBOSE=1
}

g4py-test(){
   g4py-pycheck && PYTHONPATH=$(g4py-libdir) python -c "import Geant4"
}

g4py(){
   g4py-pycheck && echo to use access g4py : import Geant4 as g4  && PYTHONPATH=$(g4py-libdir):$PYTHONPATH ipython
}


g4py-g4global(){
   cd $(g4py-g4-bdir)/source
   make CLHEP_BASE_DIR=$(g4py-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1  global
}

g4py-g4global-libs-(){ cat << EOL
G4persistency
G4parmodels
G4global
G4geometry
G4particles
G4processes
G4physicslists
G4interfaces
G4digits_hits
EOL
}

g4py-g4global-install(){
     local msg="$FUNCNAME :"
     cd $(g4py-g4-bdir)/source
     local lib
     local libpath
     local dest=$(g4py-g4-idir)/lib
     [ ! -d "$dest" ] && echo $msg destination dir missing $dest && return 1
     g4py-g4global-libs- | while read lib ; do
        libpath=../lib/Linux-g++/lib$lib.so
        [ ! -f "$libpath" ] && echo $msg missing library $libpath && return 1
        [ -f "$libpath" ] && echo $msg cp $libpath $dest && cp $libpath $dest 
     done
}



