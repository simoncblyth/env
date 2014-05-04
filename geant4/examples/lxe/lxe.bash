# === func-gen- : geant4/examples/lxe fgp geant4/examples/lxe.bash fgn lxe fgh geant4/examples
lxe-src(){      echo geant4/examples/lxe/lxe.bash ; }
lxe-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lxe-src)} ; }
lxe-vi(){       vi $(lxe-source) ; }
lxe-env(){      

    nuwa-;
    elocal- ; 
    fenv ;   ## CRUCIAL STEP OF SETTING UP ENV CORRESPONDING TO DYB INSTALL ARE USING 

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




EOU
}
lxe-dir(){  echo $(nuwa-g4-bdir)/examples/extended/optical/LXe ; }
lxe-sdir(){ echo $(env-home)/geant4/examples/lxe ; }
lxe-cd(){  cd $(lxe-dir); }
lxe-scd(){ cd $(lxe-sdir); }

lxe-make(){
   lxe-cd
   lxe-customize
   make CPPVERBOSE=1 CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 XERCESCROOT=$(nuwa-xercesc-idir) $*
}

lxe-bin(){ echo $(lxe-dir)/../../../../bin/Linux-g++/LXe ; }

lxe-run(){
   lxe-cd
   local cmd="$(lxe-bin) $*"
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
   local name=${1:-LXeStackingAction}

   local hdr=$(lxe-sdir)/include/$name.hh 
   local imp=$(lxe-sdir)/src/$name.cc 

   cp $hdr $(lxe-dir)/include/
   [ -f "$imp" ] && cp $imp $(lxe-dir)/src/
}

lxe-customize(){
  lxe-place LXeStackingAction
  lxe-place ChromaPhotonList
  cp $(lxe-sdir)/GNUmakefile $(lxe-dir)/  
}

lxe-grab-chromaphotonlist(){
   chromaserver-
   cp $(chromaserver-dir)/src/ChromaPhotonList.hh $(lxe-sdir)/include/
}

