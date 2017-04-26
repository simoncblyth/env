gdml-src(){      echo geant4/geometry/gdml/gdml.bash ; }
gdml-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gdml-src)} ; }
gdml-vi(){       vi $(gdml-source) ; }
gdml-usage(){ cat << EOU

GDML testing
=============



Refs
-----

* http://gdml.web.cern.ch/GDML/
* http://gdml.web.cern.ch/GDML/doc/GDMLmanual.pdf

* ~/opticks_refs/GDMLmanual.pdf 


Releases
-----------

* 

::

    02/06/2016   GDML_3_1_4 released, Updated User's Manual, release 2.6 
    09/10/2015   GDML_3_1_3 released 
    12/06/2015   GDML_3_1_2 released, Updated User's Manual, release 2.5 
    18/11/2014   GDML_3_1_1 released, Updated User's Manual, release 2.4 
    26/11/2013   GDML_3_1_0 released, Updated User's Manual, release 2.3 
    23/11/2011   GDML_3_0_1 released, Updated User's Manual, release 2.2 
    15/12/2010   Updated User's Manual, release 2.1 
    18/12/2008   GDML_3_0_0 released 





Build/Install issues
---------------------

* similar to those reported in *g4py-*

  * due to lack of global libs by default
  * also lack of XercesC

N xercesc_2_7 xercesc_2_8 mixup 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

System xerces-c 2.7 getting linked
unintentionally for libG4persistency.so at least.::

    [blyth@belle7 source]$ gdml-checklib 
        0   322 ../lib/Linux-g++/libG4DAE.so 
      317     0 ../lib/Linux-g++/libG4persistency.so 

After forcibly rebuild/reinstall persistency::

    [blyth@belle7 source]$ find persistency -name '*.cc' -exec touch {} \;
    [blyth@belle7 source]$ gdml-build-g4-global
            ...

    [blyth@belle7 source]$ gdml-checklib
        0   322 ../lib/Linux-g++/libG4DAE.so 
        0   302 ../lib/Linux-g++/libG4persistency.so 

    [blyth@belle7 source]$ gdml-install-g4-libs

::

    /data1/env/local/dybx/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/libG4persistency.so: undefined reference to `xercesc_2_7::XMLString::transcode(char const*, unsigned short*, unsigned int, xercesc_2_7::MemoryManager*)'
    /data1/env/local/dybx/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/libG4persistency.so: undefined reference to `xercesc_2_7::XMLString::transcode(unsigned short const*)'
    /data1/env/local/dybx/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/libG4persistency.so: undefined reference to `xercesc_2_7::XercesDOMParser::setErrorHandler(xercesc_2_7::ErrorHandler*)'

But expecting::

    [blyth@belle7 gdml]$ nm $(gdml-xercesc-libdir)/libxerces-c.so | c++filt | grep xercesc | grep XMLString  | head -3
    0029ccc0 T xercesc_2_8::XMLScanner::setURIStringPool(xercesc_2_8::XMLStringPool*)
    002d0180 W xercesc_2_8::XSNamedMap<xercesc_2_8::XSIDCDefinition>::XSNamedMap(unsigned int, unsigned int, xercesc_2_8::XMLStringPool*, bool, xercesc_2_8::MemoryManager*)
    002c9600 W xercesc_2_8::XSNamedMap<xercesc_2_8::XSObject>::XSNamedMap(unsigned int, unsigned int, xercesc_2_8::XMLStringPool*, bool, xercesc_2_8::MemoryManager*)


::

    [blyth@belle7 ~]$ nm /data1/env/local/dybx/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/libG4persistency.so | c++filt | grep xercesc_2_7  | wc -l
    317
    [blyth@belle7 ~]$ nm /data1/env/local/dybx/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/libG4persistency.so | c++filt | grep xercesc_2_8  | wc -l
    0


D : chroma misses G4GDMLParser.hh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tis in tarball::

    chroma-cd src
    tar ztvf geant4.9.5.p01.tar.gz | grep GDMLParser

See *chroma-deps-rebuild-geant4*

D: architecture mixup
~~~~~~~~~~~~~~~~~~~~~~

::

    ld: warning: ld: warning: ignoring file gdmltest.o, file was built for unsupported file format ( 0xCF 0xFA 0xED 0xFE 0x07 0x00 0x00 0x01 0x03 0x00 0x00 0x00 0x01 0x00 0x00 0x00 ) which is not the architecture being linked (i386): gdmltest.oignoring file /opt/local/lib/libxerces-c.dylib, file was built for x86_64 which is not the architecture being linked (i386): /opt/local/lib/libxerces-c.dylib



EOU
}
gdml-env(){      
   elocal- 
   #case $NODE_TAG in 
   #  D) chroma- ;;
   #  *) nuwa- ;;
   #esac
}


gdml-pdf(){ open ~/opticks_refs/GDMLmanual.pdf  ; }

gdml-dir(){ echo $(gdml-g4-bdir)/source/persistency/gdml ; }
gdml-sdir(){ echo $(env-home)/geant4/geometry/gdml ; }
gdml-cd(){  cd $(gdml-dir); }
gdml-scd(){  cd $(gdml-sdir); }
gdml-mate(){ mate $(gdml-dir) ; }
gdml-get(){
   local dir=$(dirname $(gdml-dir)) &&  mkdir -p $dir && cd $dir

}

gdml-build(){
   cd $(gdml-g4-bdir)/source/persistency/gdml
   make CLHEP_BASE_DIR=$(gdml-clhep-incdir) G4SYSTEM=$(gdml-g4-system) G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(gdml-xercesc-incdir)
}

gdml-build-persistency(){
   cd $(gdml-g4-bdir)/source/persistency



   make CLHEP_BASE_DIR=$(gdml-clhep-incdir) G4SYSTEM=$(gdml-g4-system) G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(gdml-xercescroot) 
   make CLHEP_BASE_DIR=$(gdml-clhep-idir) G4SYSTEM=$(gdml-g4-system) G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(gdml-xercescroot) global
}


gdml-build-g4-global(){
   cd $(gdml-g4-bdir)/source
   #find persistency -name '*.cc' -exec touch {} \;
   make CLHEP_BASE_DIR=$(gdml-clhep-idir) G4SYSTEM=$(gdml-g4-system) G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(gdml-xercescroot) global
}

gdml-libsymcount(){
   echo $(nm $1 | c++filt | grep -c $2) 
}

gdml-checklib(){
   cd $(gdml-g4-bdir)/source
   local libext=so
   local libpath
   ls -1 ../lib/$(gdml-g4-system)/*.$libext | while read libpath ;  do
      local nm27=$(gdml-libsymcount $libpath xercesc_2_7)
      local nm28=$(gdml-libsymcount $libpath xercesc_2_8)
      if [ "$nm27" != "0" -o "$nm28" != "0" ]; then
         printf "%5s %5s %s \n" $nm27 $nm28 $libpath 
      fi 
   done
}


gdml-install-g4-libs(){
   cd $(gdml-g4-bdir)/source
   local libext=so
   local line
   ls -1 ../lib/$(gdml-g4-system)/*.$libext | while read line ;  do
       local libpath=$line
       local libname=$(basename $libpath)
       local libdest=$(gdml-g4-libdir)/$libname
       #echo ... $libname ... $libpath  ... $libdest ...

       if [ "$libpath" -nt "$libdest" ]; then
           printf "%10s %s \n" INSTALL $libdest 
           local cmd="cp $libpath $libdest"
           eval $cmd
       else
           printf "%10s %s \n" ASIS $libdest 
       fi  
   done
}


 
gdml-install(){
   cd $(nuwa-g4-bdir)/source/persistency/gdml
   cp ../../../lib/Linux-g++/libG4gdml.so $(nuwa-g4-libdir)/
   cp include/* $(nuwa-g4-incdir)/
   cp include/* ../../../include/

   # no install target 
   #make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) install

}

gdml-install-persistency(){
   cd $(gdml-g4-bdir)/source/persistency
   cp ../../lib/$(gdml-g4-system)/libG4persistency.so $(gdml-g4-libdir)/
}


gdml-g4-system(){
  case $NODE_TAG in 
    D) echo Darwin-UNSUPPORTED ;;
    *) echo Linux-g++ ;;
  esac
}

gdml-g4-bdir(){ 
  case $NODE_TAG in 
    D) echo $(chroma-g4-bdir) ;;
    N) echo $(DYB=x nuwa-g4-bdir) ;;
    *) echo $(nuwa-g4-bdir) ;;
  esac
}

gdml-g4-incdir(){ 
  case $NODE_TAG in 
    D) echo $(chroma-g4-incdir) ;;
    N) echo $(DYB=x nuwa-g4-incdir) ;;
    *) echo $(nuwa-g4-incdir) ;;
  esac
}
gdml-g4-libdir(){ 
  case $NODE_TAG in 
    D) echo $(chroma-g4-libdir) ;;
    N) echo $(DYB=x nuwa-g4-libdir) ;;
    *) echo $(nuwa-g4-libdir) ;;
  esac
}

gdml-clhep-idir(){ echo $(dirname $(gdml-clhep-incdir)); }
gdml-clhep-incdir(){ 
  case $NODE_TAG in 
    D) echo $(chroma-clhep-incdir) ;;
    N) echo $(DYB=x nuwa-clhep-incdir) ;;
    *) echo $(nuwa-clhep-incdir) ;;
  esac
}
gdml-clhep-libdir(){ 
  case $NODE_TAG in 
    D) echo $(chroma-clhep-libdir) ;;
    N) echo $(DYB=x nuwa-clhep-libdir) ;;
    *) echo $(nuwa-clhep-libdir) ;;
  esac
}
gdml-clhep-lib(){ 
  case $NODE_TAG in 
    *) echo CLHEP ;;
  esac 
}

gdml-xercescroot(){
  case $NODE_TAG in 
    *) echo $(dirname $(gdml-xercesc-incdir)) ;;
  esac
} 
gdml-xercesc-incdir(){ 
  case $NODE_TAG in 
    D) echo $(chroma-xercesc-incdir) ;;
    N) echo $(DYB=x nuwa-xercesc-incdir) ;;
    *) echo $(nuwa-xercesc-incdir) ;;
  esac
}
gdml-xercesc-libdir(){ 
  case $NODE_TAG in 
    D) echo $(chroma-xercesc-libdir) ;;
    N) echo $(DYB=x nuwa-xercesc-libdir) ;;
    *) echo $(nuwa-xercesc-libdir) ;;
  esac
}




gdml-exename(){ echo ${GDML_EXENAME:-gdmltest} ; }
gdml-exepath(){ echo $(local-base)/env/geant4/geometry/gdml/$(gdml-exename) ; }
gdml-exepath-run(){
   local exepath=$(gdml-exepath)
   exec $exepath $*
}

gdml-test(){
   type $FUNCNAME
   cd $(env-home)/geant4/geometry/gdml

   local name=$(gdml-exename)

   # if omit the xercesc incdir the system xerces-c gets used causing linker problems later
   g++ -c -I$(gdml-g4-incdir) \
          -I$(gdml-clhep-incdir) \
          -I$(gdml-xercesc-incdir) \
           -DG4LIB_USE_GDML \
        $name.cc -o $name.o

   local exepath=$(gdml-exepath)
   mkdir -p $(dirname $exepath)

   #local opt="-m32"
   local opt=""

   g++ $opt $name.o -o $exepath \
        -L$(gdml-xercesc-libdir) -lxerces-c  \
        -L$(gdml-g4-libdir) \
           -lG4persistency \
           -lG4readout \
           -lG4run \
           -lG4event \
           -lG4tracking \
           -lG4parmodels \
           -lG4processes \
           -lG4digits_hits \
           -lG4track \
           -lG4particles \
           -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms \
           -lG4interfaces -lG4global -lG4physicslists  \
           -lG4FR \
           -lG4visHepRep \
           -lG4RayTracer \
           -lG4VRML \
           -lG4Tree \
           -lG4modeling \
             -lm

    rm $name.o
   
    cat << EOS

   SKIPPED THESE

           -lG4Tree \
      #     -lG4OpenGL \
      #     -lG4vis_management \
           -lG4modeling \
      #    -L$(gdml-clhep-libdir) -l$(gdml-clhep-lib) \
             -lm
 

EOS
 

}

gdml-example-g01(){
    cd $(nuwa-g4-bdir)/examples/extended/persistency/gdml/G01
    make CLHEP_BASE_DIR=$(nuwa-clhep-idir)  G4SYSTEM=Linux-g++ XERCESCROOT=$(nuwa-xercesc-idir) CPPVERBOSE=1 G4INSTALL=../../../../.. 
}



