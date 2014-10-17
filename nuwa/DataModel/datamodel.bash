# === func-gen- : nuwa/DataModel/datamodel fgp nuwa/DataModel/datamodel.bash fgn datamodel fgh nuwa/DataModel
datamodel-src(){      echo nuwa/DataModel/datamodel.bash ; }
datamodel-source(){   echo ${BASH_SOURCE:-$(env-home)/$(datamodel-src)} ; }
datamodel-vi(){       vi $(datamodel-source) ; }
datamodel-env(){      elocal- ; chroma- ; }
datamodel-usage(){ cat << EOU

NuWa DataModel Extracts
========================

Attempt to extract and compile a primitive 
subset of NuWa DataModel (GOD derived) to allow standalone tests. 


Build Issues
-------------

* commented stream for ../SimEvent/src/SimVertex.cc


FUNCTIONS
----------

*datamodel-get*
      copy DataModel sources and modify for limited dependency operation  

*datamodel-test-all*
      test compile each SimEvent/Event header in a generated cpp


EOU
}
datamodel-prefix(){ echo $(local-base)/env/nuwa ; }
datamodel-dir(){ echo $(local-base)/env/nuwa/DataModel ; }
datamodel-sdir(){ echo $(env-home)/nuwa/DataModel ; }
datamodel-tmpdir(){ echo /tmp/env/nuwa/DataModel ; }

datamodel-cd(){  cd $(datamodel-dir); }
datamodel-scd(){  cd $(datamodel-sdir); }
datamodel-tcd(){  cd $(datamodel-tmpdir); }

datamodel-node(){ echo C ; }
datamodel-dybdir(){ 
  case $(datamodel-node) in 
    N) echo /data1/env/local/dyb ;;
    C) echo /data/env/local/dyb/trunk ;; 
  esac
}

datamodel-get(){
   $FUNCNAME-prep
   $FUNCNAME-basis
   $FUNCNAME-gaudikernel
   $FUNCNAME-unboost
   $FUNCNAME-unhepmc
   $FUNCNAME-addvector
}

datamodel-get-prep(){
   local dir=$(datamodel-dir) &&  mkdir -p $dir 
}

datamodel-get-basis(){
   datamodel-cd

   local node=$(datamodel-node)
   local moddir=$(datamodel-dybdir)/NuWa-trunk/dybgaudi/DataModel
   local hdds="SimEvent/Event Conventions/Conventions BaseEvent/Event Context/Context"

   for hdd in $hdds ; do
      if [ ! -d "$hdd" ]; then 
         echo getting hdd $hdd
         mkdir -p $hdd
         rsync -av --exclude=".svn*" --exclude="*/.svn*" --exclude="*/*/.svn*" --exclude="*.obj2doth" --include="*.h$" $node:$moddir/$hdd/ $hdd/
      else
         echo hdd $hdd exists already 
      fi 
   done

   local srcs="SimEvent/src Conventions/src BaseEvent/src Context/src"
   for src in $srcs ; do
      if [ ! -d "$src" ]; then 
         echo getting src $src
         mkdir -p $src
         rsync -av --exclude=".svn*" --exclude="*/.svn*" --include="*.cc$" $node:$moddir/$src/ $src/
      else
         echo src $src exists already 
      fi 
   done
}



datamodel-get-gaudikernel-rels(){ cat << EOR
GaudiKernel/Point3DTypes.h 
GaudiKernel/Vector3DTypes.h 
GaudiKernel/GenericAddress.h 
GaudiKernel/DataObject.h 
GaudiKernel/Kernel.h 
GaudiKernel/IOpaqueAddress.h
GaudiKernel/ClassID.h
GaudiKernel/StatusCode.h
GaudiKernel/IssueSeverity.h
GaudiKernel/IRegistry.h
GaudiKernel/SystemOfUnits.h
src/Lib/DataObject.cpp
src/Lib/LinkManager.cpp 
GaudiKernel/StreamBuffer.h
GaudiKernel/swab.h
GaudiKernel/LinkManager.h
GaudiKernel/IInspector.h
GaudiKernel/IInterface.h
EOR
} 


datamodel-get-gaudikernel(){
   local iwd=$PWD
   datamodel-cd
   local gkdir=$(datamodel-dybdir)/NuWa-trunk/gaudi/GaudiKernel
   local node=$(datamodel-node)
   local rel
   $FUNCNAME-rels | while read rel ; do 
       local urel=GaudiKernel/$rel
       echo $urel 
       if [ ! -f "$urel" ]; then  
           mkdir -p $(dirname $urel)
           local cmd="scp ${node}:$gkdir/$rel $urel"
           echo $cmd
           eval $cmd        
       else
           echo already have $urel
       fi  
    done
    cd $iwd
}


datamodel-get-dybkernel-rels(){ cat << EOR
DybKernel/IRegistrationSequence.h
src/IRegistrationSequence.cc
DybKernel/ObjectReg.h
src/ObjectReg.cc
EOR
} 

datamodel-get-dybkernel(){
   local iwd=$PWD
   datamodel-cd
   local dkdir=$(datamodel-dybdir)/NuWa-trunk/dybgaudi/DybKernel
   local node=$(datamodel-node)
   local rel
   $FUNCNAME-rels | while read rel ; do 
       local urel=DybKernel/$rel
       echo $urel 
       if [ ! -f "$urel" ]; then  
           mkdir -p $(dirname $urel)
           local cmd="scp ${node}:$dkdir/$rel $urel"
           echo $cmd
           eval $cmd        
       else
           echo already have $urel
       fi  
    done
    cd $iwd
}




datamodel-get-unboost(){
   datamodel-cd
   local hdr
   find . -name '*.h' | while read hdr ; do
       echo $hdr
       perl -pi -e 's,#include "GaudiKernel/boost_allocator.h",//#include "GaudiKernel/boost_allocator.h",' $hdr
       perl -pi -e 's,#include "GaudiKernel/SerializeSTL.h",//#include "GaudiKernel/SerializeSTL.h",' $hdr
       perl -pi -e 's,using GaudiUtils::operator<<;,//using GaudiUtils::operator<<;,' $hdr
   done
}

datamodel-get-unhepmc(){
   datamodel-cd
   local hdr
   find . -name '*.h' | while read hdr ; do
       echo $hdr
       perl -pi -e 's,#include "HepMC/GenEvent.h",//#include "HepMC/GenEvent.h",' $hdr
       perl -pi -e 's,HepMC::GenParticle,void,' $hdr
   done
}

datamodel-get-addvector(){
   datamodel-cd
   local hdr=SimEvent/Event/SimHitHeader.h
   [ "$(grep -c '#include <vector>' $hdr)" == "0" ] &&  perl -pi -e 's,#include <map>,#include <map>\n#include <vector>,' $hdr  
}

datamodel-get-wingodnoalloc(){
   datamodel-cd
   local hdr=DybKernel/DybKernel/ObjectReg.h
   [ "$(grep -c '_WIN32' $hdr)" == "1" ] &&  perl -pi -e 's,_WIN32,GOD_NOALLOC,' $hdr  
}



datamodel-test-all(){
  datamodel-cd
  local path
  ls -1 SimEvent/Event/*.h | while read path ; do 
       local hdr=$(basename $path)
       echo datamodel-test ${hdr/.h}   
       datamodel-test ${hdr/.h}   
       [ $? -ne 0 ] && return 1
  done
}

datamodel-test-(){ cat << EOT
// $FUNCNAME
#include "Event/${1:-SimHit}.h"

EOT
}

datamodel-test(){

   local iwd=$PWD
   local name=${1:-SimHit}
   local tmpd=$(datamodel-tmpdir)



   mkdir -p $tmpd
   cd $tmpd

   $FUNCNAME- $name > $name.cpp
   datamodel-compile $name.cpp -c

   cd $iwd 
}



datamodel-compile(){
   local cpp=$1
   shift 
   local dmd=$(datamodel-dir)

   local cmd="clang $* -I$dmd/Context -I$dmd/BaseEvent -I$dmd/Conventions -I$dmd/SimEvent -I$dmd/GaudiKernel -I$(chroma-root-incdir) -I$clhep -DGOD_NOALLOC $cpp "
   echo $cmd PWD $PWD
   eval $cmd
   [ $? -ne 0 ] && return 1
}


datamodel-link(){  
   local cpp=$1
   shift 
   local dmd=$(datamodel-dir)
   local clhep=$(dirname $(chroma-clhep-incdir))   # G4 incdir,risky 

   clang $* -I$dmd/Context \
            -I$dmd/BaseEvent \
            -I$dmd/Conventions \
            -I$dmd/SimEvent \
            -I$dmd/GaudiKernel \
            -I$(chroma-root-incdir) \
            -I$clhep \
            -DGOD_NOALLOC \
             $cpp 
}


 
datamodel-cmake(){
   mkdir -p $(datamodel-tmpdir)
   datamodel-tcd
   cmake $(datamodel-sdir) -DCMAKE_INSTALL_PREFIX=$(datamodel-prefix)
}
datamodel-make(){
   datamodel-tcd
   make $*
}
datamodel-install(){
   datamodel-make install
}
datamodel-build(){
   datamodel-cmake
   datamodel-make
   datamodel-install
}

