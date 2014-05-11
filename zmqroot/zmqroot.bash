# === func-gen- : zmqroot/zmqroot fgp zmqroot/zmqroot.bash fgn zmqroot fgh zmqroot
zmqroot-src(){      echo zmqroot/zmqroot.bash ; }
zmqroot-source(){   echo ${BASH_SOURCE:-$(env-home)/$(zmqroot-src)} ; }
zmqroot-vi(){       vi $(zmqroot-source) ; }
zmqroot-usage(){ cat << EOU

ZMQROOT
=======

Simple C++ class for sending/receiving 
ROOT TObject derived objects using ZEROMQ.


FUNCTIONS
----------

::

    zmqroot-build
    zmqroot-build-full   # deletes build directory first 


ROOT TObject Serialize/Deserialize
-----------------------------------

* https://halldweb1.jlab.org/wiki/index.php/CMsgRootConsumer.cc



NuWa/CMT Integration
-----------------------

* http://dayabay.ihep.ac.cn/tracs/dybsvn/changeset/3734

Dictionary creation problems

TObject 



EOU
}

zmqroot-dir(){  echo $(local-base)/env/zmqroot ; }
zmqroot-bdir(){ echo /tmp/env/zmqroot-build ; }
zmqroot-sdir(){ echo $(env-home)/zmqroot ; }

zmqroot-cd(){   cd $(zmqroot-dir); }
zmqroot-scd(){  cd $(zmqroot-sdir); }
zmqroot-bcd(){  cd $(zmqroot-bdir); }

zmqroot-env(){      
   elocal-

    # TODO: get rid of envvar requirements 
   zeromq-
   chroma-
   chroma-geant4-export  
}

zmqroot-prefix(){ echo $(zmqroot-dir) ; }
zmqroot-libraries(){    echo ZMQRoot ; }
zmqroot-lib(){  echo $(zmqroot-prefix)/lib/libZMQRoot.dylib ;}
zmqroot-export(){
  export ZMQROOT_PREFIX=$(zmqroot-prefix)
  export ZMQROOT_LIBRARIES="$(zmqroot-libraries)"
  export ZMQROOT_LIB="$(zmqroot-lib)"
}

zmqroot-verbose(){ echo 1 ; }

zmqroot-cmake(){
   mkdir -p $(zmqroot-bdir)   
   zmqroot-bcd
   cmake -DCMAKE_INSTALL_PREFIX=$(zmqroot-dir) $(zmqroot-sdir) 
}
zmqroot-make(){
   zmqroot-bcd
   make VERBOSE=$(zmqroot-verbose) 
}
zmqroot-install(){
   zmqroot-bcd
   make install VERBOSE=$(zmqroot-verbose)
}
zmqroot-wipe(){
   local msg="=== $FUNCNAME :"
   local bdir="$(zmqroot-bdir)"
   echo $msg deleting bdir $bdir
   rm -rf "$bdir"
}
zmqroot-build(){
  zmqroot-cmake
  zmqroot-make
  zmqroot-install
}
zmqroot-build-full(){
  zmqroot-wipe
  zmqroot-build
}

zmqroot-otool(){
  otool-
  otool-info $(zmqroot-lib)
}





zmqroot-nuwapkg(){    echo $DYB/NuWa-trunk/dybgaudi/Utilities/ZMQRoot ; }
zmqroot-nuwapkg-cd(){ cd $(zmqroot-nuwapkg) ; }
zmqroot-nuwapkg-cpto(){

   local pkg=$(zmqroot-nuwapkg) 
   local nam=$(basename $pkg)
   local inc=$pkg/$nam
   local src=$pkg/src
   local dict=$pkg/dict
  
   local iwd=$PWD 
   zmqroot-scd

   cp include/ZMQRoot.hh    $inc/
   cp include/MyTMessage.hh $inc/
   cp include/MyTMessage_LinkDef.h $dict/

   cp src/ZMQRoot.cc        $src/
   perl -pi -e 's,ZMQRoot.hh,ZMQRoot/ZMQRoot.hh,' $src/ZMQRoot.cc 
   perl -pi -e 's,MyTMessage.hh,ZMQRoot/MyTMessage.hh,' $src/ZMQRoot.cc 

   cp src/MyTMessage.cc     $src/
   perl -pi -e 's,MyTMessage.hh,ZMQRoot/MyTMessage.hh,' $src/MyTMessage.cc 


   cd $iwd
}


