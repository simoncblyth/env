
lxe-clhep-idir(){
   case $NODE_TAG in 
     N) echo $(nuwa-clhep-idir) ;; 
     D) echo $(chroma-clhep-prefix)/include ;; 
   esac
}
lxe-xercesc-idir(){
   case $NODE_TAG in 
     N) echo $(nuwa-xercesc-idir) ;; 
     D) echo $(xercesc-prefix)/include ;; 
   esac
}
lxe-system(){
   case $NODE_TAG in 
     N) echo Linux-g++ ;;
     D) echo Darwin-clang ;;
  esac
}
lxe-g4-bdir(){
   case $NODE_TAG in 
     D) echo  ;;
     *) echo $(nuwa-g4-bdir) ;; 
   esac
}

lxe-rootcint-deprecated(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   lxe-cd include
   echo $msg from $PWD

   local line
   local kls
   local cmd
   ls -1 *_LinkDef.h | while read line ; do
      kls=${line/_LinkDef.h}
      cmd="rootcint -v -f ../src/${kls}Dict.cc -c -p -I$(lxe-g4-bdir)/include -I$(lxe-clhep-idir)/include ${kls}.hh ${kls}_LinkDef.h"
      echo $msg $cmd 
      eval $cmd
   done  

   cd $iwd
}

lxe-gmake-deprecated(){
   lxe-cd
   lxe-customize
   lxe-rootcint-deprecated
   make CPPVERBOSE=1 CLHEP_BASE_DIR=$(lxe-clhep-idir) G4SYSTEM=$(lxe-system) G4LIB_BUILD_SHARED=1 XERCESCROOT=$(lxe-xercesc-idir) $*
}



