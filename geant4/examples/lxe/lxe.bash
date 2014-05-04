# === func-gen- : geant4/examples/lxe fgp geant4/examples/lxe.bash fgn lxe fgh geant4/examples
lxe-src(){      echo geant4/examples/lxe/lxe.bash ; }
lxe-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lxe-src)} ; }
lxe-vi(){       local dir=$(dirname $(lxe-source)); cd $dir ; vi lxe.bash; }
lxe-env(){      

    nuwa-;
    elocal- ; 
    fenv ;   ## CRUCIAL STEP OF SETTING UP ENV CORRESPONDING TO DYB INSTALL ARE USING 

    zeromq-

}
lxe-usage(){ cat << EOU

GEANT4 LXE EXAMPLE
===================

BUILDING
----------

::

    lxe-make
    lxe-make clean
    lxe-make bin -n    # to see the commands and locate the binary 


RUNNING
----------

Issues:

#. vis drivers not built ? OGLSX fails


ChromaPhotonList 
------------------



EOU
}
lxe-dir(){  echo $(nuwa-g4-bdir)/examples/extended/optical/LXe ; }
lxe-sdir(){ echo $(env-home)/geant4/examples/lxe ; }
lxe-cd(){  cd $(lxe-dir)/$1; }
lxe-scd(){ cd $(lxe-sdir); }

lxe-make(){
   lxe-cd
   lxe-customize
   lxe-rootcint

   make CPPVERBOSE=1 CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 XERCESCROOT=$(nuwa-xercesc-idir) $*
}

lxe-host(){ echo localhost ; }
lxe-port(){ echo 5555 ; }
lxe-config(){
   export LXE_CLIENT_CONFIG="tcp://$(lxe-host):$(lxe-port)"
}

lxe-bin(){ echo $(lxe-dir)/../../../../bin/Linux-g++/LXe ; }

lxe-run(){
   lxe-cd

   lxe-config
   env | grep LXE

   local cmd="LD_LIBRARY_PATH=${ZEROMQ_PREFIX}/lib:$LD_LIBRARY_PATH $(lxe-bin) $*"
   echo $cmd
   eval $cmd 
}

lxe-test(){ lxe-run $(lxe-sdir)/test.mac ; }

lxe-grab(){
   local name=${1:-LXeStackingAction}
   lxe-scd
   cp $(lxe-dir)/include/$name.hh include/
   cp $(lxe-dir)/src/$name.cc src/
}

lxe-place(){
   local msg="=== $FUNCNAME :"
   local name=${1:-LXeStackingAction}

   echo $msg $name

   local hdr=$(lxe-sdir)/include/$name.hh 
   local imp=$(lxe-sdir)/src/$name.cc 

   cp $hdr $(lxe-dir)/include/
   [ -f "$imp" ] && cp $imp $(lxe-dir)/src/
}

lxe-customize(){
  lxe-place LXeStackingAction
  lxe-place ChromaPhotonList
  lxe-place MyTMessage

  cp $(lxe-sdir)/GNUmakefile $(lxe-dir)/  
  cp $(lxe-sdir)/include/ChromaPhotonList_LinkDef.h $(lxe-dir)/include/
  cp $(lxe-sdir)/include/MyTMessage_LinkDef.h $(lxe-dir)/include/

}

lxe-rootcint(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   lxe-cd include
   echo $msg from $PWD

   local line
   local kls
   local cmd
   ls -1 *_LinkDef.h | while read line ; do
      kls=${line/_LinkDef.h}
      cmd="rootcint -v -f ../src/${kls}Dict.cc -c -p -I../../../../../include -I$(nuwa-clhep-idir)/include ${kls}.hh ${kls}_LinkDef.h"
      echo $msg $cmd 
      eval $cmd
   done  

   cd $iwd
}


lxe-grab-chromaphotonlist(){
   chromaserver-
   cp $(chromaserver-dir)/src/ChromaPhotonList.hh $(lxe-sdir)/include/
}

