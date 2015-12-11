# === func-gen- : geant4/examples/lxe fgp geant4/examples/lxe.bash fgn lxe fgh geant4/examples
lxe-src(){      echo geant4/examples/lxe/lxe.bash ; }
lxe-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lxe-src)} ; }
lxe-vi(){       local dir=$(dirname $(lxe-source)); cd $dir ; vi lxe.bash; }
lxe-builder(){
  case $NODE_TAG in 
     N) echo gmake ;; 
     D) echo cmake ;; 
  esac
}

lxe-usage(){ cat << EOU

GEANT4 LXE EXAMPLE
===================


Looks like, I complicated this with ChromaPhotonList and ZMQ based server...


Docs
----

* http://geant4.web.cern.ch/geant4/UserDocumentation/Doxygen/examples_doc/html/ExampleLXe.html

Unusual SD:

    The PMT sensitive detector cannot be triggered like a normal sensitive detector
    because the sensitive volume does not allow photons to pass through it. Rather,
    it detects them in the OpBoundary process based on an efficiency set on the
    skin of the volume.

    A normal sensitive detector would have its ProcessHits function called for each
    step by a particle inside the volume. So, to record these hits with a sensitive
    detector we watched the status of the OpBoundary process from the stepping
    manager whenever a photon hit the sensitive volume of the pmt. If the status
    was 'Detection', we retrieve the sensitive detector from G4SDManager and call
    its ProcessHits function.


CMAKE BUILDING AGAINST CHROMA G4
----------------------------------

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/ch03s02.html
* http://geant4.web.cern.ch/geant4/UserDocumentation/Doxygen/examples_doc/html/README_HowToRun.html


DYBX xercesc issue on N
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trying to build the example with DYBX geant4 in order to export LXe geometry with G4DAE.

Before adding XERCECS hookup to GNUmakefile link errors from both xercesc_2_8 for G4DAE and xercesc_2_7 for G4gdml::

    man     -lCLHEP -lm
    ../../../../lib/Linux-g++/libG4DAE.so: undefined reference to `xercesc_2_8::XMLEntityDecl::~XMLEntityDecl()'
    ../../../../lib/Linux-g++/libG4DAE.so: undefined reference to `xercesc_2_8::XMLString::release(unsigned short**)'
    ../../../../lib/Linux-g++/libG4gdml.so: undefined reference to `xercesc_2_7::XMLString::transcode(char const*, unsigned short*, unsigned int, xercesc_2_7::MemoryManager*)'
    ../../../../lib/Linux-g++/libG4DAE.so: undefined reference to `xercesc_2_8::XMemory::operator new(unsigned int)'
    ../../../../lib/Linux-g++/libG4gdml.so: undefined reference to `xercesc_2_7::XMLString::transcode(unsigned short const*)'
    ../../../../lib/Linux-g++/libG4gdml.so: undefined reference to `xercesc_2_7::XercesDOMParser::setErrorHandler(xercesc_2_7::ErrorHandler*)'

After, only get link error with xercesc_2_7 G4gdml::

    pecsolids -lG4digits -lG4csg -lG4hepnumerics -lG4bosons -lG4cuts -lG4navigation -lG4volumes -lG4procman -lG4track -lG4magneticfield -lG4geometrymng -lG4materials -lG4partman -lG4graphics_reps -lG4intercoms -lG4globman     -lCLHEP -lm
    ../../../../lib/Linux-g++/libG4gdml.so: undefined reference to `xercesc_2_7::XMLString::transcode(char const*, unsigned short*, unsigned int, xercesc_2_7::MemoryManager*)'
    ../../../../lib/Linux-g++/libG4gdml.so: undefined reference to `xercesc_2_7::XMLString::transcode(unsigned short const*)'
    ../../../../lib/Linux-g++/libG4gdml.so: undefined reference to `xercesc_2_7::XercesDOMParser::setErrorHandler(xercesc_2_7::ErrorHandler*)'

Why two versions  ?::

    [blyth@belle7 LXe]$ ll ../../../../lib/Linux-g++/libG4gdml.so
    -rwxrwxr-x 1 blyth blyth 6663749 Mar  6 14:36 ../../../../lib/Linux-g++/libG4gdml.so
    [blyth@belle7 LXe]$ ll ../../../../lib/Linux-g++/libG4DAE.so
    -rwxrwxr-x 1 blyth blyth 669175 Mar  7 19:24 ../../../../lib/Linux-g++/libG4DAE.so
    [blyth@belle7 LXe]$ 

Interference from system libxercesc-c ?::

    [blyth@belle7 LXe]$ ll /usr/lib/libxerces-c.so.27
    lrwxrwxrwx 1 root root 19 Sep 30  2013 /usr/lib/libxerces-c.so.27 -> libxerces-c.so.27.0

Opening can of worms with -L/usr/lib -lxercec-c fails to work 
(maybe because the 2_8 lib of that name was already loaded)
but explicitly providing the path to system 2_7 lib works::

     LDFLAGS += /usr/lib/libxerces-c.so.27



cmake RPATH ISSUE 
------------------



::

    (chroma_env)delta:LXe-build blyth$ ln -s /usr/local/env/zmqroot/lib/libZMQRoot.dylib libZMQRoot.dylib
    (chroma_env)delta:LXe-build blyth$ ./LXe 




RUNNING
----------

vis drivers
~~~~~~~~~~~~~~~~

#. vis drivers not built ? OGLSX fails



EOU
}
lxe-name(){ echo LXe ; }
lxe-dir(){  
  case $NODE_TAG in 
    N) echo $(nuwa-g4-bdir)/examples/extended/optical/LXe ;;
    D) echo $VIRTUAL_ENV/src/geant4.9.5.p01/examples/extended/optical/LXe
  esac
}

lxe-sdir(){ echo $(env-home)/geant4/examples/lxe ; }
lxe-cd(){  cd $(lxe-dir)/$1; }
lxe-scd(){ cd $(lxe-sdir); }

# cmake build dir
lxe-bdir(){  echo $(lxe-dir)-build ; }    
lxe-bcd(){   mkdir -p $(lxe-bdir) && cd $(lxe-bdir); }
lxe-bwipe(){ rm -rf $(lxe-bdir); }


lxe-env(){      
    elocal- ; 
    case $NODE_TAG in 
       N) lxe-env-N ;;
       D) lxe-env-D ;;
       *) echo NO Geant4 INSTALL on $NODE_TAG ;; 
    esac
    zmqroot-
    cpl-
}
lxe-env-D(){
    chroma-
}
lxe-env-N(){
    nuwa-
    nuwa-export-prefix

    if [ "$DYB" == "$DYBX" ]; then 
        dyb-- dybdbi
    else 
        fenv ;      # fast way only applicable for standard DYB
    fi 
}




lxe-cmake(){
   local msg="=== $FUNCNAME :"

   chroma-geant4-export
   lxe-customize

   lxe-bcd
   echo $msg $PWD
   local cmd="cmake  -DGeant4_DIR=$(chroma-geant4-dir) $(lxe-dir) "
   echo $msg $cmd
   eval $cmd
}
lxe-make(){
   lxe-bcd
   make $(lxe-name) VERBOSE=1
}
lxe-build(){
   lxe-cmake 
   lxe-make 
}
lxe-build-full(){
   lxe-bwipe   # DELETE BUILDDIR TO FORCE FULL BUILD
   lxe-build
}
lxe-otool(){
   otool-
   otool-info $(lxe-bin)
}


############################################

lxe-host(){ echo localhost ; }
lxe-port(){ echo 5555 ; }
lxe-config(){
   export LXE_CLIENT_CONFIG="tcp://$(lxe-host):$(lxe-port)"
   env | grep LXE
}

lxe-bin(){ 
   case $NODE_TAG in 
     N) echo $(lxe-dir)/../../../../bin/Linux-g++/LXe ;;
     D) echo $(lxe-bdir)/LXe ;;
   esac
}

lxe-libpathvar(){
   case $(uname) in 
     Linux)  echo LD_LIBRARY_PATH ;;
     Darwin) echo DYLD_LIBRARY_PATH ;;
   esac
}

lxe-run-manual(){
   zmqroot-export
   cpl-export
   local libpathvar=$(lxe-libpathvar)
   local cmd="$libpathvar=${ZMQROOT_PREFIX}/lib:${CHROMAPHOTONLIST_PREFIX}/lib $(lxe-bin) $*"
   echo $cmd
   eval $cmd 
}
lxe-run-rpath(){
   local cmd="$(lxe-bin) $*"
   echo $cmd
   eval $cmd 
}

lxe-run(){ 
   lxe-cd
   lxe-config
   case $NODE_TAG in 
     N) lxe-run-manual $* ;;
     D) lxe-run-rpath $* ;;
     *) lxe-run-rpath $* ;;
   esac
}

lxe-test(){ lxe-run $(lxe-sdir)/test.mac ; }


########### copy code from SVN into the geant4 example source directory #########

lxe-cp(){
   local msg="=== $FUNCNAME :"
   local nam
   for nam in $* ; do
      local src=$(lxe-sdir)/$nam
      local tgt=$(lxe-dir)/$nam

      echo $msg nam $nam src $src tgt $tgt

      [ ! -f "$tgt" ] && cp $src $tgt 
      [ "$src" -nt "$tgt" ] && echo $msg $src $tgt && cp $src $tgt 
      #[ "$tgt" -nt "$src" ] && echo $msg $src $tgt SKIP
   done
}

lxe-customize(){
  case $(lxe-builder) in 
   gmake) lxe-customize-gmake ;;
   cmake) lxe-customize-cmake ;;
       *) echo NO BUILDER && sleep 1000000000 ;;
  esac

  local klss="LXeStackingAction ChromaPhotonList"
  local kls
  for kls in $klss ; do 
     lxe-cp include/$kls.hh src/$kls.cc
  done
  lxe-cp LXe.cc test.mac include/ChromaPhotonList_LinkDef.h 
}

lxe-customize-cmake(){ lxe-cp CMakeLists.txt ;}
lxe-customize-make(){  lxe-cp GNUmakefile    ;}


########################################






lxe-grab(){
   local name=${1:-LXeStackingAction}
   lxe-scd
   cp $(lxe-dir)/include/$name.hh include/
   cp $(lxe-dir)/src/$name.cc src/
}


lxe-grab-chromaphotonlist(){
   chromaserver-
   cp $(chromaserver-dir)/src/ChromaPhotonList.hh $(lxe-sdir)/include/
}


