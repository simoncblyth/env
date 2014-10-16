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


PyZMQ Serialization
---------------------

* http://zeromq.github.io/pyzmq/serialization.html

TMessage Compression
----------------------

* http://personalpages.to.infn.it/~puccio/htmldoc/AliHLTMessage.h
* http://web.ift.uib.no/~kjeks/doc/alice-hlt-current/AliHLTMessage_8cxx_source.html
* http://root.cern.ch/root/html/TSocket.html
* http://root.cern.ch/root/html/src/TSocket.cxx.html#zXkgI

Zero Copy
----------

* https://gist.github.com/GaelVaroquaux/1249305

  Copy-less bindings of C-generated arrays with Cython





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

zmqroot-icd(){   cd $(zmqroot-dir); }
zmqroot-cd(){   cd $(zmqroot-sdir); }
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

zmqroot-setup-test(){
  export PATH=$(zmqroot-dir)/bin:$PATH 
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

zmqroot-nuwapkg(){
  case $NODE_TAG in
     N) echo $DYB/NuWa-trunk/dybgaudi/Utilities/ZMQRoot ;;
     *) utilities- && echo $(utilities-dir)/ZMQRoot ;;
  esac
}
zmqroot-nuwapkg-cd(){ cd $(zmqroot-nuwapkg)/$1 ; }


zmqroot-nuwapkg-cpto-cmds(){

   local pkg=$(zmqroot-nuwapkg) 
   local nam=$(basename $pkg)
   local inc=$pkg/$nam
   local src=$pkg/src
   local dict=$pkg/dict
     
   cat << EOC
   cp ZMQRoot/ZMQRoot.hh    $inc/
   cp ZMQRoot/MyTMessage.hh $inc/
   cp dict/MyTMessage_LinkDef.h $dict/

   cp src/ZMQRoot.cc        $src/
   cp src/MyTMessage.cc     $src/
EOC

}

zmqroot-old(){ cat << EOC
   perl -pi -e 's,ZMQRoot.hh,ZMQRoot/ZMQRoot.hh,' $src/ZMQRoot.cc 
   perl -pi -e 's,MyTMessage.hh,ZMQRoot/MyTMessage.hh,' $src/ZMQRoot.cc 
   perl -pi -e 's,MyTMessage.hh,ZMQRoot/MyTMessage.hh,' $src/MyTMessage.cc 
EOC
}


zmqroot-nuwapkg-cpto(){
   local iwd=$PWD 
   zmqroot-scd
   $FUNCNAME-cmds | while read cmd ; do 
      echo $cmd
      eval $cmd
   done 
   cd $iwd
}


zmqroot-nuwapkg-diff-cmds(){
   local pkg=$(zmqroot-nuwapkg)
   local pkn=$(basename $pkg)
   local sdir=$(zmqroot-sdir)
   cat << EOC
diff $sdir/ZMQRoot/ZMQRoot.hh $pkg/$pkn/ZMQRoot.hh
diff $sdir/ZMQRoot/MyTMessage.hh $pkg/$pkn/MyTMessage.hh
diff $sdir/dict/MyTMessage_LinkDef.h $pkg/dict/MyTMessage_LinkDef.h
diff $sdir/src/ZMQRoot.cc $pkg/src/ZMQRoot.cc
diff $sdir/src/MyTMessage.cc $pkg/src/MyTMessage.cc
EOC
}
zmqroot-nuwapkg-diff(){
   local cmd 
   $FUNCNAME-cmds | while read cmd ; do 
      echo $cmd
      eval $cmd
   done 
}

