# === func-gen- : chroma/G4DAEChroma/gdc fgp chroma/G4DAEChroma/gdc.bash fgn gdc fgh chroma/G4DAEChroma
gdc-src(){      echo chroma/G4DAEChroma/gdc.bash ; }
gdc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gdc-src)} ; }
gdc-vi(){       vi $(gdc-source) ; }
gdc-usage(){ cat << EOU

G4DAEChroma
============


Issues
-------

Installation tied to Chroma Geant4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Improve generality of build by simplifications 
to CMakeLists.txt

* moving things into env/cmake/Modules/ 


G4Rep3x3 not available in older G4 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    #CMT---> compiling ../src/G4DAETransformCache.cc
    ../src/G4DAETransformCache.cc: In member function 'G4AffineTransform* G4DAETransformCache::GetTransform(size_t)':
    ../src/G4DAETransformCache.cc:220: error: 'G4Rep3x3' was not declared in this scope
    ../src/G4DAETransformCache.cc:220: error: expected `;' before 'r33'

::

     40 #include "globals.hh"
     41 #include "G4ThreeVector.hh"
     42 #include <CLHEP/Vector/Rotation.h>
     43 
     44 typedef CLHEP::HepRotation G4RotationMatrix;
     45 
     46 #endif

::

     39 #include "globals.hh"
     40 #include "G4ThreeVector.hh"
     41 #include <CLHEP/Vector/Rotation.h>
     42 
     43 typedef CLHEP::HepRotation G4RotationMatrix;
     44 typedef CLHEP::HepRep3x3 G4Rep3x3;
     45 
     46 #endif



NuWa integration
~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 dybgaudi]$ svn ci Simulation Utilities -m  "minor: update Utilities/G4DAEChroma with G4DAETransformCache and cnpy dependency, specialize G4DAEGeometry in DybG4DAEGeometry rejig Chroma config in DsChromaRunAction " 
    Sending        Simulation/DetSimChroma/src/DsChromaRunAction.cc
    Sending        Simulation/DetSimChroma/src/DsChromaRunAction.h
    Adding         Simulation/DetSimChroma/src/DybG4DAEGeometry.cc
    Adding         Simulation/DetSimChroma/src/DybG4DAEGeometry.h
    Adding         Utilities/G4DAEChroma/G4DAEChroma/DemoG4DAECollector.hh
    Sending        Utilities/G4DAEChroma/G4DAEChroma/G4DAEChroma.hh
    Adding         Utilities/G4DAEChroma/G4DAEChroma/G4DAECommon.hh
    Sending        Utilities/G4DAEChroma/G4DAEChroma/G4DAEGeometry.hh
    Sending        Utilities/G4DAEChroma/G4DAEChroma/G4DAEHit.hh
    Adding         Utilities/G4DAEChroma/G4DAEChroma/G4DAETransformCache.hh
    Sending        Utilities/G4DAEChroma/G4DAEChroma/G4DAETransport.hh
    Sending        Utilities/G4DAEChroma/cmt/requirements
    Adding         Utilities/G4DAEChroma/src/DemoG4DAECollector.cc
    Sending        Utilities/G4DAEChroma/src/G4DAEChroma.cc
    Sending        Utilities/G4DAEChroma/src/G4DAECollector.cc
    Adding         Utilities/G4DAEChroma/src/G4DAECommon.cc
    Sending        Utilities/G4DAEChroma/src/G4DAEGeometry.cc
    Sending        Utilities/G4DAEChroma/src/G4DAEHit.cc
    Sending        Utilities/G4DAEChroma/src/G4DAESensDet.cc
    Adding         Utilities/G4DAEChroma/src/G4DAETransformCache.cc
    Sending        Utilities/G4DAEChroma/src/G4DAETransport.cc
    Adding         Utilities/cnpy
    Adding         Utilities/cnpy/LICENSE
    Adding         Utilities/cnpy/README
    Adding         Utilities/cnpy/cmt
    Adding         Utilities/cnpy/cmt/requirements
    Adding         Utilities/cnpy/cmt/version.cmt
    Adding         Utilities/cnpy/cnpy
    Adding         Utilities/cnpy/cnpy/cnpy.h
    Adding         Utilities/cnpy/src
    Adding         Utilities/cnpy/src/cnpy.cpp
    Transmitting file data ...........................
    Committed revision 23488.
    [blyth@belle7 dybgaudi]$ e


Wrote transform cache to::

    scp -r CN:/data1/env/local/env/muon_simulation/optical_photon_weighting/OPW/DybG4DAEGeometry.cache .

::

    In [1]: import numpy as np

    In [2]: a = np.load("DybG4DAEGeometry.cache/data.npy")

    In [3]: len(a)
    Out[3]: 684

    In [4]: a[0]
    Out[4]: 
    array([[  5.67843745e-01,   8.23136369e-01,  -6.95385058e-17,
              6.68898507e+05],
           [  8.23136369e-01,  -5.67843745e-01,  -1.00801803e-16,
             -4.39222475e+05],
           [ -1.22460635e-16,   6.16297582e-33,  -1.00000000e+00,
             -4.96150000e+03],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              1.00000000e+00]])

    In [5]: k = np.load("DybG4DAEGeometry.cache/key.npy")

    In [6]: len(k)
    Out[6]: 684

    In [7]: k
    Out[7]: 
    array([16842753, 16842754, 16842755, 16842756, 16842757, 16842758,
           16843009, 16843010, 16843011, 16843012, 16843013, 16843014,




EOU
}

gdc-name(){  echo G4DAEChroma ; }
gdc-dir(){   echo $(local-base)/env/chroma/$(gdc-name) ; }
gdc-prefix(){ echo $(gdc-dir) ; }

gdc-sdir(){ echo $(env-home)/chroma/$(gdc-name) ; }
gdc-bdir(){ echo /tmp/env/chroma/$(gdc-name) ; }

gdc-cd(){  cd $(gdc-sdir); }
gdc-icd(){  cd $(gdc-dir); }
gdc-scd(){  cd $(gdc-sdir); }
gdc-bcd(){  cd $(gdc-bdir); }


gdc-geant4-home(){
  case $NODE_TAG in
    D) echo /usr/local/env/chroma_env/src/geant4.9.5.p01 ;;
  esac
}
gdc-geant4-dir(){   
  case $NODE_TAG in
    D) echo /usr/local/env/chroma_env/lib/Geant4-9.5.1 ;;
  esac
}

gdc-env(){      
   elocal- ; 
   export GEANT4_HOME=$(gdc-geant4-home)

   rootsys-  # needed to find root headers
}

gdc-cmake(){
   type $FUNCNAME
   local iwd=$PWD
   mkdir -p $(gdc-bdir)
   gdc-bcd
   cmake -DGeant4_DIR=$(gdc-geant4-dir) \
         -DCMAKE_INSTALL_PREFIX=$(gdc-prefix) \
         -DCMAKE_BUILD_TYPE=Debug \
         $(gdc-sdir)

   cd $iwd
}
gdc-verbose(){ echo  1 ; }
gdc-make(){
   local iwd=$PWD
   gdc-bcd
   make $* VERBOSE=$(gdc-verbose)
   cd $iwd
}
gdc-install(){ gdc-make install ; }

gdc-build(){
   gdc-cmake
   #gdc-make
   gdc-install
}
gdc--(){ 
   gdc-install 
}

gdc-build-full(){
   gdc-wipe
   gdc-build
}
gdc-wipe(){
   rm -rf $(gdc-bdir)
}


############## running 

gdc-exe(){ echo $(gdc-prefix)/bin/$(gdc-name)Test ; }
gdc-run-env(){
  export-
  export-export
  #env | grep DAE
}
gdc-run(){
  $FUNCNAME-env
  local exe=$(gdc-exe)
  ls -l $exe 
  local cmd="$exe $*"
  echo $cmd
  eval $cmd
}


############## over to NuWa

gdc-nuwapkg(){    
  if [ -n "$DYB" ]; then 
     echo $DYB/NuWa-trunk/dybgaudi/Utilities/$(gdc-name) 
  else
     utilities- && echo $(utilities-dir)/$(gdc-name) 
  fi
}
  
gdc-nuwapkg-cd(){ cd $(gdc-nuwapkg)/$1 ; } 


gdc-names(){ 
   local path
   ls -1 $(gdc-sdir)/G4DAEChroma/*.hh  | while read path ; do
      local name=$(basename $path)
      echo ${name/.hh}
   done
}

gdc-extra-hdrs(){ cat << EOX
numpy.hpp
md5digest.h
EOX
}


gdc-nuwapkg-action-cmds(){
   local action=${1:-diff}
   local pkg=$(gdc-nuwapkg)
   local pkn=$(basename $pkg)
   local nam=$(gdc-name)
   local sdir=$(gdc-sdir)

   cat << EOC
mkdir -p $pkg/$pkn
mkdir -p $pkg/src
EOC

   gdc-names | while read nam ; do

   if [ "$action" == "cpfr" ]; then

   cat << EOC
cp $pkg/$pkn/$nam.hh $sdir/$pkn/$nam.hh         
cp $pkg/src/$nam.cc $sdir/src/$nam.cc          
EOC
 
   else

   cat << EOC
$action $sdir/$pkn/$nam.hh         $pkg/$pkn/$nam.hh
$action $sdir/src/$nam.cc          $pkg/src/$nam.cc
EOC
   fi  


   done

   local hdr
   gdc-extra-hdrs | while read hdr ; do

   if [ "$action" == "cpfr" ]; then

   cat << EOC
cp $pkg/$pkn/$hdr $sdir/$pkn/$hdr         
EOC
 
   else 

   cat << EOC
$action $sdir/$pkn/$hdr         $pkg/$pkn/$hdr
EOC
 
   fi

   done



}

gdc-nuwapkg-action(){
   local cmd
   $FUNCNAME-cmds $1 | while read cmd ; do
      echo $cmd
      eval $cmd
   done
}
gdc-nuwapkg-diff(){  gdc-nuwapkg-action diff ; }
gdc-nuwapkg-cpto(){  gdc-nuwapkg-action cp ; }
gdc-nuwapkg-cpfr(){  gdc-nuwapkg-action cpfr ; }



gdc-nuwacfg () 
{ 
    local msg="=== $FUNCNAME :";
    local pkg=$1;
    shift;
    [ ! -d "$pkg/cmt" ] && echo ERROR pkg $pkg has no cmt dir && sleep 1000000;
    local iwd=$PWD;
    echo $msg for pkg $pkg;
    cd $pkg/cmt;
    cmt config;
    . setup.sh;
    cd $iwd
}



gdc-nuwaenv()
{   
    zmqroot-
    gdc-nuwacfg $(zmqroot-nuwapkg);
    cpl-
    gdc-nuwacfg $(cpl-nuwapkg);
    csa-
    gdc-nuwacfg $(csa-nuwapkg)
}

gdc-nuwapkg-make() 
{ 
    local iwd=$PWD;

    dyb-;
    dyb-setup;

    #gdc-nuwaenv

    gdc-nuwapkg-cd cmt

    cmt br cmt config
    cmt config
    cmt make

    cd $iwd
}


gdc-nuwapkg-prerequisites()
{
    [ -z "$DYB" ] && echo DYB is not defined && return 1
    cd $DYB

    ./dybinst trunk external zmq



    zmqroot-                    # TO BE REMOVED
    zmqroot-nuwapkg-make

    cpl-                        # ChromaPhotonList   TO BE REMOVED
    cpl-nuwapkg-make

    cnpy-                       # TO BE REMOVED
    cnpy-nuwapkg-make


}

gdc-config-edit(){ vi $(gdc-config-json) ; }
gdc-config-check(){ python -c "import json, pprint ; print pprint.pformat(json.load(file('$(gdc-config-json)')))" ; } 
gdc-config-json(){ echo $(gdc-sdir)/config.json ; }
gdc-flags-json(){  echo $(gdc-sdir)/flags.json ; }
gdc-flags-gen(){ PYTHONPATH=$HOME flags.py $(gdc-flags-json) ; }

