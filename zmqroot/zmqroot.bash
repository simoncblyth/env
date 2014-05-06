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


LIBRARY USAGE ISSUE
--------------------

::

    (chroma_env)delta:LXe-build blyth$ ./LXe 
    dyld: Library not loaded: libZMQRoot.dylib
      Referenced from: /usr/local/env/chroma_env/src/geant4.9.5.p01/examples/extended/optical/LXe-build/./LXe
      Reason: image not found

OSX 10.9.2 LD_LIBRARY_PATH not working::

    (chroma_env)delta:LXe-build blyth$ LD_LIBRARY_PATH=/usr/local/env/zmqroot/lib ./LXe
    dyld: Library not loaded: libZMQRoot.dylib
      Referenced from: /usr/local/env/chroma_env/src/geant4.9.5.p01/examples/extended/optical/LXe-build/./LXe
      Reason: image not found
    Trace/BPT trap: 5

Need DYLD_LIBRARY_PATH::

    (chroma_env)delta:LXe-build blyth$ DYLD_LIBRARY_PATH=/usr/local/env/zmqroot/lib ./LXe

    *************************************************************
     Geant4 version Name: geant4-09-05-patch-01    (20-March-2012)
    ...

All other libs have a path, rather than just a name::

    (chroma_env)delta:LXe-build blyth$ otool -L LXe | grep libZMQRoot
        libZMQRoot.dylib (compatibility version 0.0.0, current version 0.0.0)



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
zmqroot-export(){
  export ZMQROOT_PREFIX=$(zmqroot-prefix)
  export ZMQROOT_LIBRARIES="$(zmqroot-libraries)"
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


