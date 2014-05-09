# === func-gen- : chroma/ChromaZMQRootTest/czrt fgp chroma/ChromaZMQRootTest/czrt.bash fgn czrt fgh chroma/ChromaZMQRootTest
czrt-src(){      echo chroma/ChromaZMQRootTest/czrt.bash ; }
czrt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(czrt-src)} ; }
czrt-vi(){       vi $(czrt-source) ; }
czrt-env(){      elocal- ; }
czrt-usage(){ cat << EOU

Chroma ZMQRoot Test 
=====================

Memory Issue
--------------

With MyTMessage used on stack in the test (rather than heap) 
get **pointer being freed was not allocated**

#. TODO: check for leaks

::

    (chroma_env)delta:ChromaZMQRootTest blyth$ lldb $(czrt-bin)
    Current executable set to '/usr/local/env/chroma/ChromaZMQRootTest/bin/ChromaZMQRootTest' (x86_64).
    (lldb) r
    Process 36786 launched: '/usr/local/env/chroma/ChromaZMQRootTest/bin/ChromaZMQRootTest' (x86_64)
    ZMQRoot::ZMQRoot envvar [CHROMA_CLIENT_CONFIG] config [tcp://localhost:5555] 
    ChromaPhotonList::Print  [1]
    ZMQRoot::SendObject sent bytes: 217 
    ZMQRoot::ReceiveObject received bytes: 217 
    ZMQRoot::ReceiveObject reading TObject from the TMessage 
    ZMQRoot::ReceiveObject returning TObject 
    ChromaZMQRootTest(36786,0x7fff7ab15310) malloc: *** error for object 0x10561df08: pointer being freed was not allocated
    *** set a breakpoint in malloc_error_break to debug
    Process 36786 stopped
    * thread #1: tid = 0x278fe8, 0x00007fff8c34e866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8c34e866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8c34e866:  jae    0x7fff8c34e870            ; __pthread_kill + 20
       0x7fff8c34e868:  movq   %rax, %rdi
       0x7fff8c34e86b:  jmpq   0x7fff8c34b175            ; cerror_nocancel
       0x7fff8c34e870:  ret    

    (lldb) bt
    * thread #1: tid = 0x278fe8, 0x00007fff8c34e866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8c34e866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8d5f835c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff9320ab1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff903a607f libsystem_malloc.dylib`free + 411
        frame #4: 0x0000000100066f07 libCore.so`TBuffer::~TBuffer() + 39
        frame #5: 0x0000000100043155 libZMQRoot.dylib`MyTMessage::~MyTMessage() + 21
        frame #6: 0x0000000100043135 libZMQRoot.dylib`MyTMessage::~MyTMessage() + 21
        frame #7: 0x0000000100044eb5 libZMQRoot.dylib`ZMQRoot::ReceiveObject() + 469
        frame #8: 0x0000000100005e02 ChromaZMQRootTest`main + 370



EOU
}
czrt-cd(){  cd $(czrt-dir); }
czrt-mate(){ mate $(czrt-dir) ; }
czrt-get(){
   local dir=$(dirname $(czrt-dir)) &&  mkdir -p $dir && cd $dir

}
czrt-name(){ echo ChromaZMQRootTest ; }
czrt-dir(){  echo $(local-base)/env/chroma/$(czrt-name) ; }
czrt-sdir(){ echo $(env-home)/chroma/$(czrt-name) ; }
czrt-bdir(){ echo /tmp/env/chroma/$(czrt-name) ; }

czrt-bin(){ echo $(czrt-prefix)/bin/$(czrt-name) ; }

czrt-cd(){   cd $(czrt-dir); }
czrt-scd(){  cd $(czrt-sdir); }
czrt-bcd(){  cd $(czrt-bdir); }

czrt-verbose(){ echo 1 ; }
czrt-prefix(){ echo $(czrt-dir) ; }

czrt-geant4-home(){ 
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/src/geant4.9.5.p01 ;;
  esac
}
czrt-geant4-dir(){ 
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/lib/Geant4-9.5.1 ;;
  esac
}
czrt-rootsys(){
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/src/root-v5.34.14 ;;
  esac
}


czrt-env(){      
   elocal- 
   export GEANT4_HOME=$(czrt-geant4-home)
   export ROOTSYS=$(czrt-rootsys)   # needed to find rootcint for dictionary creation
}


czrt-wipe(){
   local msg="=== $FUNCNAME :"
   local bdir="$(czrt-bdir)"
   echo $msg deleting bdir $bdir
   rm -rf "$bdir"
}
czrt-cmake(){
   type $FUNCNAME
   local iwd=$PWD
   mkdir -p $(czrt-bdir)   
   czrt-bcd
   cmake 
         -DCMAKE_INSTALL_PREFIX=$(czrt-prefix) \
         $(czrt-sdir) 

   cd $iwd
}
czrt-make(){
   local iwd=$PWD
   czrt-bcd
   make $* VERBOSE=$(czrt-verbose) 
   cd $iwd
}
czrt-install(){
   czrt-make install
}
czrt-build(){
   czrt-cmake
   czrt-make
   czrt-install
}
czrt-build-full(){
   czrt-wipe
   czrt-build
}
czrt-otool(){
   otool-
   otool-info $(czrt-lib)
}

czrt-nuwapkg(){    echo $DYB/NuWa-trunk/dybgaudi/Utilities/$(czrt-name) ; }  
czrt-nuwapkg-cd(){ cd $(czrt-nuwapkg) ; }
czrt-nuwapkg-cpto(){

   local pkg=$(czrt-nuwapkg)   
   local nam=$(basename $pkg)
   local inc=$pkg/$nam
   local src=$pkg/src

   mkdir -p $pkg
   mkdir -p $pkg/src
  
   local iwd=$PWD 
   czrt-scd

   local main=ChromaZMQRootTest.cc
   cp $main    $src/

   perl -pi -e 's,ChromaPhotonList.hh,Chroma/ChromaPhotonList.hh,' $src/$main
   perl -pi -e 's,ZMQRoot.hh,ZMQRoot/ZMQRoot.hh,'                  $src/$main


   cd $iwd
}






