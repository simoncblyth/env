# === func-gen- : nuwa/detsim/csa fgp nuwa/detsim/csa.bash fgn csa fgh nuwa/detsim
csa-src(){      echo nuwa/detsim/csa.bash ; }
csa-source(){   echo ${BASH_SOURCE:-$(env-home)/$(csa-src)} ; }
csa-vi(){       vi $(csa-source) ; }
csa-env(){      elocal- ; }
csa-usage(){ cat << EOU

ChromaStackAction
==================

Running
---------

::

    delta:~ blyth$ ssh N
    Last login: Sun May 11 14:41:39 2014 from simon.phys.ntu.edu.tw
    [blyth@belle7 ~]$ csa.sh


GDB
----

Attach debugger using::

    csa-;csa-nuwarun-gdb $(pgrep -f nuwa.py)

* NB use single quote, and hit TAB for completion

::

    (gdb) b 'DsChromaStackAction::CollectPhoton(G4Track const*)

    Breakpoint 1, DsChromaStackAction::CollectPhoton (this=0xbce3ec0, aPhoton=0x11b67160) at ../src/DsChromaStackAction.cc:86
    86     G4ParticleDefinition* pd = aPhoton->GetDefinition();
    (gdb) bt
    #0  DsChromaStackAction::CollectPhoton (this=0xbce3ec0, aPhoton=0x11b67160) at ../src/DsChromaStackAction.cc:86
    #1  0x06c9349f in DsChromaStackAction::ClassifyNewTrack (this=0xbce3ec0, aTrack=0x11b67160) at ../src/DsChromaStackAction.cc:175
    #2  0x0800fa60 in G4StackManager::PushOneTrack (this=0xbc33b38, newTrack=0x11b67160, newTrajectory=0x0) at src/G4StackManager.cc:74
    #3  0x07fd1ca0 in G4EventManager::StackTracks (this=0xbc28920, trackVector=0xbc28388, IDhasAlreadySet=false) at src/G4EventManager.cc:293
    #4  0x07fd24ee in G4EventManager::DoProcessing (this=0xbc28920, anEvent=0xfaacbc0) at src/G4EventManager.cc:232
    #5  0x07fd29e6 in G4EventManager::ProcessOneEvent (this=0xbc28920, anEvent=0xfaacbc0) at src/G4EventManager.cc:335
    #6  0xb486b5e8 in GiGaRunManager::processTheEvent (this=0xbc28088) at ../src/component/GiGaRunManager.cpp:207
    ...
    (gdb) p this
    $1 = (DsChromaStackAction * const) 0xbce3ec0
    (gdb) p this->fPhotonList
    $2 = (class ChromaPhotonList *) 0xbce7920
    (gdb) p this->fPhotonList->x.size()
    $3 = 825574
    (gdb) del 1
    (gdb) p this->fPhotonList->x.size()
    $4 = 825575
    (gdb) c
    Continuing.
    Program received signal SIGINT, Interrupt.
    G4DynamicParticle::GetTotalMomentum (this=0x11b76ab8) at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/particles/management/include/G4DynamicParticle.icc:185
    185 inline G4double G4DynamicParticle::GetTotalMomentum() const
    (gdb) bt
    #0  G4DynamicParticle::GetTotalMomentum (this=0x11b76ab8) at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/particles/management/include/G4DynamicParticle.icc:185
    #1  0xb4a2df32 in DsG4OpBoundaryProcess::PostStepDoIt (this=0xb776540, aTrack=@0x11b76f60, aStep=@0xbc29250) at ../src/DsG4OpBoundaryProcess.cc:158
    #2  0x07188e1a in G4SteppingManager::InvokePSDIP (this=0xbc29138, np=3) at src/G4SteppingManager2.cc:517
    #3  0x07189113 in G4SteppingManager::InvokePostStepDoItProcs (this=0xbc29138) at src/G4SteppingManager2.cc:493
    #4  0x07184f4c in G4SteppingManager::Stepping (this=0xbc29138) at src/G4SteppingManager.cc:210
    #5  0x0719350a in G4TrackingManager::ProcessOneTrack (this=0xbc29110, apValueG4Track=0x11b76f60) at src/G4TrackingManager.cc:126
    #6  0x07fd224f in G4EventManager::DoProcessing (this=0xbc28920, anEvent=0xfaacbc0) at src/G4EventManager.cc:185
    #7  0x07fd29e6 in G4EventManager::ProcessOneEvent (this=0xbc28920, anEvent=0xfaacbc0) at src/G4EventManager.cc:335
    #8  0xb486b5e8 in GiGaRunManager::processTheEvent (this=0xbc28088) at ../src/component/GiGaRunManager.cpp:207
    ...
    (gdb) q
    The program is running.  Quit anyway (and detach it)? (y or n) y
    Detaching from program: /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python, process 20816
    [blyth@belle7 ~]$ 



EOU
}
csa-dir(){ echo $(env-home)/nuwa/detsim ; }
csa-cd(){  cd $(csa-dir); }
csa-mate(){ mate $(csa-dir) ; }
csa-get(){
   local dir=$(dirname $(csa-dir)) &&  mkdir -p $dir && cd $dir

}

csa-nuwapkg(){ echo $DYB/NuWa-trunk/dybgaudi/Simulation/DetSimChroma ; }

csa-nuwapkg-cd(){ cd $(csa-nuwapkg)/$1 ; }
csa-nuwapkg-cpto(){ 
   local iwd=$PWD 
   local pkg=$(csa-nuwapkg)
   local nam=DsChromaStackAction

   csa-cd

   cp src/$nam.h  $pkg/src/
   cp src/$nam.cc $pkg/src/

   perl -pi -e 's,ChromaPhotonList.hh,Chroma/ChromaPhotonList.hh,' $pkg/src/$nam.cc
   perl -pi -e 's,ZMQRoot.hh,ZMQRoot/ZMQRoot.hh,'                  $pkg/src/$nam.cc

   cd $iwd
}   
csa-nuwapkg-diff(){
   local pkg=$(csa-nuwapkg)
   local nam=DsChromaStackAction

   diff $(csa-dir)/src/$nam.h   $pkg/src/$nam.h
   diff $(csa-dir)/src/$nam.cc  $pkg/src/$nam.cc
}


csa-nuwapkg-make(){
   local iwd=$PWD

   csa-nuwaenv

   csa-nuwapkg-cd cmt
   cmt config
   cmt make 

   cd $iwd
}



csa-nuwacfg(){
   local msg="=== $FUNCNAME :"
   local pkg=$1
   shift  # protect cmt from args
   [ ! -d "$pkg/cmt" ] && echo ERROR NO cmt SUBDIR && sleep 1000000
   local iwd=$PWD

   echo $msg for pkg $pkg
   cd $pkg/cmt

   cmt config
   . setup.sh 

   cd $iwd
}

csa-nuwaenv(){

   opw-       # opw-env sets up NuWa env 

   zmqroot-
   csa-nuwacfg $(zmqroot-nuwapkg)

   cpl- 
   csa-nuwacfg $(cpl-nuwapkg)

   csa-
   csa-nuwacfg $(csa-nuwapkg)

}

csa-nuwarun-pid(){ echo $(pgrep -f nuwa.py) ; }
csa-nuwarun-gdb(){
   
   local def=$(csa-nuwarun-pid)
   local pid=${1:-$def}
   [ -z $pid ] && echo enter pid of nuwa.py process && return 1
   opw-
   gdb $(which python) $pid
}



csa-nuwarun(){


   csa-nuwaenv
   opw-cd     # need to be in OPW to find "fmcpmuon"


   zmq-
   export CSA_CLIENT_CONFIG=$(zmq-broker-url)     # override default set in requirements

   #nuwa.py -n 1 -m "fmcpmuon --use-basic-physics --chroma --test"
   nuwa.py -n 100 -m "fmcpmuon --use-basic-physics --chroma "

}

csa-nuwa-send-test-cpl(){
   csa-nuwaenv
   zmq-
   CHROMA_CLIENT_CONFIG=$(zmq-broker-url) ChromaZMQRootTest.exe
}



csa-nuwarun-notes(){ cat << EON


EON
}

csa-lslib(){
   local lib=$DYB/NuWa-trunk/dybgaudi/InstallArea/$CMTCONFIG/lib
   ls -l $lib/libChroma* $lib/libZMQ* $lib/libDetSimChroma*

}
