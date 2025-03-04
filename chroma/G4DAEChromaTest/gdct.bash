# === func-gen- : chroma/G4DAEChromaTest/gdct fgp chroma/G4DAEChromaTest/gdct.bash fgn gdct fgh chroma/G4DAEChromaTest
gdct-src(){      echo chroma/G4DAEChromaTest/gdct.bash ; }
gdct-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gdct-src)} ; }
gdct-vi(){       vi $(gdct-source) ; }
gdct-usage(){ cat << EOU

G4DAEChromaTest
=================

Usage::

    delta:~ blyth$ gdct-;gdct-build-full

    delta:~ blyth$ which G4DAEChromaTest
    /usr/local/env/nuwa/bin/G4DAEChromaTest

    delta:~ blyth$ G4DAEChromaTest
    geokey DAE_NAME_DYB_GDML : missing : use "export-;export-export" to define  
    failed to load geometry with geokey DAE_NAME_DYB_GDML

    delta:~ blyth$ export-;export-export
    delta:~ blyth$ G4DAEChromaTest
    geokey DAE_NAME_DYB_GDML geopath /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 


Test Network Mirroring frontend/broker/backend
------------------------------------------------

Compile the p_network function in G4DAEChromaTest.cc, 
with a choice of Photons implementation.

Open three terminals:

* broker   czmq_broker.sh 
* worker   *gdct-;gdct-worker*   talks to BACKEND
* client   *gdct-;gdct-client*   talks to FRONTEND

The frontend lives for one call only, the other two are 
long lived. Note that the broker and backend mirroring 
are blind to the type, they just shovel bytes, so can 
leave them running and experiment with different 
types for the frontend.


EOU
}
gdct-prefix(){ echo $(gdct-dir) ; }
gdct-env(){      
       elocal- 
       PATH=/usr/local/env/chroma_env/bin:$PATH    
       ## put geant4-config in PATH for cmake to find G4 
}

gdct-dir(){ echo $(local-base)/env/nuwa ; }
gdct-name(){ echo G4DAEChromaTest ; }
gdct-sdir(){ echo $HOME/env/chroma/$(gdct-name); }
gdct-tdir(){ echo /tmp/env/chroma/$(gdct-name) ; }

gdct-cd(){  cd $(gdct-sdir); }


gdct-bcd(){  cd $(gdct-dir); }
gdct-tcd(){  cd $(gdct-tdir); }


gdct-cmake(){
   local iwd=$PWD
   mkdir -p $(gdct-tdir)
   gdct-tcd
   cmake $(gdct-sdir) -DCMAKE_INSTALL_PREFIX=$(gdct-prefix) -DCMAKE_BUILD_TYPE=Debug 
   cd $iwd
}
gdct-make(){
   local iwd=$PWD
   gdct-tcd
   make $*
   cd $iwd
   [ "$?" != "0" ] && echo $msg $FUNCNAME ERROR && return 1
}
gdct-install(){
   gdct-make install
}
gdct--(){
   gdct-install
}
gdct---(){
   gdct-install
   G4DAEChromaTest $*
}


gdct-build(){
   gdct-cmake
   #gdct-make
   gdct-install
}
gdct-wipe(){
   rm -rf $(gdct-tdir)
}
gdct-build-full(){
   gdct-wipe
   gdct-build
}
gdct-lldb(){
   lldb $(which $(gdct-name))
}


gdct-broker(){
   czmq-
   czmq-broker-local
}
gdct-client(){
   export-
   export-export
   FRONTEND=tcp://127.0.0.1:5001 $LLDB G4DAEChromaTest
}
gdct-worker(){
   BACKEND=tcp://127.0.0.1:5002 $LLDB G4DAEChromaTest
}

