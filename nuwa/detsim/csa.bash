# === func-gen- : nuwa/detsim/csa fgp nuwa/detsim/csa.bash fgn csa fgh nuwa/detsim
csa-src(){      echo nuwa/detsim/csa.bash ; }
csa-source(){   echo ${BASH_SOURCE:-$(env-home)/$(csa-src)} ; }
csa-vi(){       vi $(csa-source) ; }
csa-usage(){ cat << EOU

ChromaStackAction and ChromaRunAction
========================================

TODO: rename to dsc- for DetSimChroma

Running
---------

::

    delta:~ blyth$ ssh N
    Last login: Sun May 11 14:41:39 2014 from simon.phys.ntu.edu.tw
    [blyth@belle7 ~]$ csa.sh


ChromaRunAction
-----------------

Based on GaussTools exporter::

   gtc=/data1/env/local/dyb/NuWa-trunk/lhcb/Sim/GaussTools/src/Components
   cp $gtc/GiGaRunActionExport.h   DsChromaRunAction.h 
   cp $gtc/GiGaRunActionExport.cpp DsChromaRunAction.cc



EOU
}
csa-dir(){ echo $(env-home)/nuwa/detsim ; }
csa-cd(){  cd $(csa-dir); }
csa-nuwapkg(){ echo $DYB/NuWa-trunk/dybgaudi/Simulation/DetSimChroma ; }
csa-nuwapkg-cd(){ cd $(csa-nuwapkg)/$1 ; }
csa-env(){      elocal- ; mocknuwa- ; }

csa-names(){ cat << EON
DsChromaStackAction
DsChromaRunAction
DybG4DAECollector
DybG4DAEGeometry
EON
}



csa-nuwapkg-cpto-(){ 
   local iwd=$PWD 
   local pkg=$(csa-nuwapkg)
   local nam=$1
   csa-cd
   cp src/$nam.h  $pkg/src/
   cp src/$nam.cc $pkg/src/
   cd $iwd
}   
csa-nuwapkg-cpfr-(){
   local iwd=$PWD 
   local pkg=$(csa-nuwapkg)
   local nam=$1
   csa-cd
   cp $pkg/src/$nam.h src/$nam.h
   cp $pkg/src/$nam.cc src/$nam.cc
   cd $iwd
}

csa-nuwapkg-cpto(){ 
   local nam
   csa-names | while read nam ; do 
      $FUNCNAME- $nam
   done

   local pkg=$(csa-nuwapkg)
   nam="DsChromaRunAction_BeginOfRunAction.icc"
   cp $(mocknuwa-sdir)/$nam $pkg/src/$nam
}
csa-nuwapkg-cpfr(){ 
   local nam
   csa-names | while read nam ; do 
      $FUNCNAME- $nam
   done

   local pkg=$(csa-nuwapkg)
   nam="DsChromaRunAction_BeginOfRunAction.icc"
   cp $pkg/src/$nam $(mocknuwa-sdir)/$nam 
}



csa-old(){ cat << EOO
   perl -pi -e 's,ChromaPhotonList.hh,Chroma/ChromaPhotonList.hh,' $pkg/src/$nam.cc
   perl -pi -e 's,ZMQRoot.hh,ZMQRoot/ZMQRoot.hh,'                  $pkg/src/$nam.cc
EOO
}


csa-nuwapkg-diff(){
   local nam
   csa-names | while read nam ; do 
      $FUNCNAME- $nam
   done

   local pkg=$(csa-nuwapkg)
   nam="DsChromaRunAction_BeginOfRunAction.icc"
   local cmd="diff $pkg/src/$nam $(mocknuwa-sdir)/$nam"
   echo $cmd
   eval $cmd
}
csa-nuwapkg-diff-(){
   local pkg=$(csa-nuwapkg)
   local nam=${1:-DsChromaStackAction}

   local exts="h cc"
   local ext
   for ext in $exts ; do 
      local cmd="diff $(csa-dir)/src/$nam.$ext   $pkg/src/$nam.$ext"
      echo $cmd
      eval $cmd 
   done 
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
csa-dbg(){  csa-nuwarun-gdb ; }
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
   export G4DAECHROMA_CLIENT_CONFIG=$(zmq-broker-url)     # override default set in requirements

   echo $FUNCNAME 
   env | grep G4DAECHROMA

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
