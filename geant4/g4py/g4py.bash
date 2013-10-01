# === func-gen- : geant4/g4py/g4py fgp geant4/g4py/g4py.bash fgn g4py fgh geant4/g4py
g4py-src(){      echo geant4/g4py/g4py.bash ; }
g4py-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4py-src)} ; }
g4py-vi(){       vi $(g4py-source) ; }
g4py-env(){      elocal- ; }
g4py-usage(){ cat << EOU

Geant4Py
=========

The configure is plumping for system python include dir even when 
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
 

After that succeed to build and install. But somehow the system, libboost_python is stepping in::

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

Up verbosity::

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





EOU
}
g4py-dir(){ echo $DYB/external/build/LCG/geant4.9.2.p01/environments/g4py ; }
g4py-cd(){  cd $(g4py-dir); }
g4py-mate(){ mate $(g4py-dir) ; }
g4py-get(){
   local dir=$(dirname $(g4py-dir)) &&  mkdir -p $dir && cd $dir

}


g4py-prefix(){ echo $(g4py-dir) ; }
g4py-libdir(){ echo $(g4py-prefix)/lib ; }

g4py-plat(){ echo i686-slc5-gcc41-dbg ; }
g4py-g4-idir(){ echo $DYB/external/geant4/4.9.2.p01/$(g4py-plat) ; }
g4py-clhep-idir(){ echo $DYB/external/clhep/2.0.4.2/$(g4py-plat) ; } 
g4py-clhep-lib(){ echo CLHEP-2.0.4.2 ; } 
g4py-boost-idir(){ echo $DYB/external/Boost/1.38.0_python2.7/$(g4py-plat) ; } 
g4py-xercesc-idir(){ echo $DYB/external/XercesC/2.8.0/$(g4py-plat) ; }
g4py-python-idir(){ echo $DYB/external/Python/2.7/$(g4py-plat) ; }

g4py-configure(){
   cd $(g4py-dir)
   ./configure  linux \
                  --prefix=$(g4py-prefix) \
                  --libdir=$(g4py-libdir) \
                  --with-g4-incdir=$(g4py-g4-idir)/include \
                   --with-g4-libdir=$(g4py-g4-idir)/lib \
                --with-clhep-incdir=$(g4py-clhep-idir)/include \
                --with-clhep-libdir=$(g4py-clhep-idir)/lib \
                   --with-clhep-lib=$(g4py-clhep-lib) \
                --with-boost-incdir=$(g4py-boost-idir)/include/boost-1_38 \
                --with-boost-libdir=$(g4py-boost-idir)/lib \
                --with-xercesc-incdir=$(g4py-xercesc-idir)/include \
                --with-xercesc-libdir=$(g4py-xercesc-idir)/lib \
                --with-python-incdir=$(g4py-python-idir)/include/python2.7 \
                --with-python-libdir=$(g4py-python-idir)/lib \
                --with-boost-python-lib=boost_python-gcc41-mt \
                --with-extra-dir=/dev/null



}

g4py-make(){
   cd $(g4py-dir)
   fenv 
   which python   # should be same python are building against, namely NuWa python
   make
}

g4py-test(){
   fenv
   which python
   PYTHONPATH=$(g4py-libdir) python -c "import Geant4"
}

g4py-dbg(){

    g4py-cd source/track 
    g++ -m32 -Wl,-rpath,/data1/env/local/dyb/external/Boost/1.38.0_python2.7/i686-slc5-gcc41-dbg/lib:/data1/env/local/dyb/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib:/data1/env/local/dyb/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg/lib:  \
         -Wl,-soname,G4track.so -shared -o \
             G4track.so \
            pyG4Step.o  pyG4StepPoint.o  pyG4StepStatus.o  pyG4Track.o  pyG4TrackStatus.o  pymodG4track.o  \
            -L/data1/env/local/dyb/external/Boost/1.38.0_python2.7/i686-slc5-gcc41-dbg/lib -lboost_python \
            -L/data1/env/local/dyb/external/XercesC/2.8.0/i686-slc5-gcc41-dbg/lib -lxerces-c \
            -L/data1/env/local/dyb/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib -lG4persistency -lG4readout -lG4run -lG4event -lG4tracking \
                         -lG4parmodels -lG4processes -lG4digits_hits -lG4track -lG4particles -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms \
                         -lG4interfaces -lG4global -lG4physicslists  -lG4FR -lG4visHepRep -lG4RayTracer -lG4VRML -lG4Tree -lG4OpenGL -lG4vis_management -lG4modeling \
            -L/data1/env/local/dyb/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg/lib -lCLHEP-2.0.4.2


}



