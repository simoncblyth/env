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
    zmqroot-fullbuild   # deletes build directory first 


INSTALL ISSUE
--------------



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

zmqroot-cmake(){
   mkdir -p $(zmqroot-bdir)   
   zmqroot-bcd
   cmake -DCMAKE_INSTALL_PREFIX=$(zmqroot-dir) $(zmqroot-sdir) 
}

zmqroot-make(){
   zmqroot-bcd
   make VERBOSE=1 
}

zmqroot-install(){
   zmqroot-bcd
   #make install DESTDIR=$(zmqroot-dir)
   make install VERBOSE=1
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

zmqroot-fullbuild(){
  zmqroot-wipe
  zmqroot-build
}


