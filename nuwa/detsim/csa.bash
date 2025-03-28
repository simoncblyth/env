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


    [blyth@ntugrid5 ~]$ csa.sh --modulo-photon 1000


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

csa-g4materials-vi(){ vi $(csa-g4materials) ; }
csa-g4materials(){ echo $(csa-cachedir)/g4materials.json ; }

csa-cachedir(){ echo $(local-base)/env/nuwa/detsim/DetSimChroma.cache ; }
csa-cachedir-ls(){ ls -l $(csa-cachedir) ;}
csa-cachedir-cd(){ cd $(csa-cachedir) ;}
csa-cachedir-srcnode(){ echo G5 ; }
csa-cachedir-names(){ cat << EON
data.npy
key.npy
g4materials.json
EON
}
csa-cachedir-scp(){
   local cachedir=$(csa-cachedir)
   local srcnode=$(csa-cachedir-srcnode)
   mkdir -p $(dirname $cachedir)
   [ "$NODE_TAG" == "$srcnode" ] && echo DISALLOED to run this from srcnode $srcnode && return 1

   local name
   local cmd
   csa-cachedir-names | while read name ; do 
       cmd="scp $srcnode:$(NODE_TAG=$srcnode csa-cachedir)/$name $cachedir/$name"
       echo $cmd
   done
}



csa-names(){ cat << EON
DsChromaPhysConsOptical
DsChromaG4Cerenkov
DsChromaG4Scintillation
DsChromaG4OpBoundaryProcess
DsChromaG4OpRayleigh
DsChromaStackAction
DsChromaRunAction
DsChromaEventAction
DybG4DAECollector
DybG4DAEGeometry
DsChromaRunAction_BeginOfRunAction
csa
EON
}
csa-exts(){
   case $1 in 
                                     csa) echo py ;;
      DsChromaRunAction_BeginOfRunAction) echo icc ;;
                                       *) echo h cc ;;
   esac 
}
csa-nuwapkg-cmd(){
   case $1 in 
     cpto) echo cp $2 $3 ;;
     cpfr) echo cp $3 $2 ;;
     diff) echo diff $2 $3 ;;
   esac
}
csa-nuwapkg-action-(){ 
   local act=$1
   local nam=$2
   local pkg=$(csa-nuwapkg)
   local exts=$(csa-exts $nam)
   #echo $FUNCNAME act $act nam $nam
   local ext
   local cmd
   for ext in $exts ; do 
      case $ext in 
        py) cmd=$(csa-nuwapkg-cmd $act $(csa-dir)/python/DetSimChroma/$nam.$ext $pkg/python/DetSimChroma/$nam.$ext) ;;
         *) cmd=$(csa-nuwapkg-cmd $act $(csa-dir)/src/$nam.$ext $pkg/src/$nam.$ext) ;;
      esac
      echo $cmd
      eval $cmd 
   done 
}   
csa-nuwapkg-action(){ 
   local act=$1
   local nam
   csa-names | while read nam ; do 
      $FUNCNAME- $act $nam
   done
}
csa-nuwapkg-cpto(){ csa-nuwapkg-action cpto ; }
csa-nuwapkg-cpfr(){ csa-nuwapkg-action cpfr ; }
csa-nuwapkg-diff(){ csa-nuwapkg-action diff ; }



csa-nuwapkg-make-py(){ csa-nuwapkg-make DetSimChroma_python ; }
csa-nuwapkg-make(){
   local iwd=$PWD

   dyb-
   dyb-setup

   csa-nuwapkg-cd cmt
   cmt config
   cmt make $*

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




csa-envcache-cd(){ cd $(dirname $(csa-envcache)) ; }
csa-envcache(){ printf $ENVCAP_BASH_TMPL csa ; }
csa-envcache-rm(){ 
  local envcache=$(csa-envcache)
  rm $envcache
} 
csa-envcache-source(){
   local msg=" === $FUNCNAME :"
   local envcache=$(csa-envcache)
   if [ -f "$envcache" ]; then
       echo $msg sourcing envcache $envcache
       source $envcache
   else
       echo $msg MISSING FILE $envcache
   fi
}


csa-pp(){
   which python
   echo $PYTHONPATH | tr ":" "\n"
}


csa-nuwaenv-manual(){

   opw-       # opw-env sets up NuWa env 

   #zmqroot-
   #csa-nuwacfg $(zmqroot-nuwapkg)

   #cpl- 
   #csa-nuwacfg $(cpl-nuwapkg)

   cjs-
   csa-nuwacfg $(cjs-nuwapkg)

   rapsqlite-
   csa-nuwacfg $(rapsqlite-nuwapkg)

   gdc-
   csa-nuwacfg $(gdc-nuwapkg)

   csa-
   csa-nuwacfg $(csa-nuwapkg)

}



csa-nuwaenv(){
   dyb-
   dyb-config $(csa-nuwapkg)

   #
   # in response to lack of MySQLdb in focussed environment config 
   # have kludged to config the kitchensink DybRelease
   # maybe adding DybPython to dependencies for runtime use of nuwa.py 
   # can avoid that 
   #
   # BUT now that envcache is working, not much motivation for this
   # 
}

csa-db(){ echo $HOME/g4daechroma.db ; }
csa-sqlite(){ sqlite3 $(csa-db) ; }
csa-config(){
   gdc-
   gdc-config-json
}

csa-export(){
   # potentially override defaults set in requirements
   zmq-
   export G4DAECHROMA_CLIENT_CONFIG=$(zmq-broker-url)     
   export G4DAECHROMA_CACHE_DIR=$(csa-cachedir) 
   export G4DAECHROMA_DATABASE_PATH=$(csa-db)
   export G4DAECHROMA_CONFIG_PATH=$(csa-config)
   env | grep G4DAECHROMA

   export-
   export-export    
   # for path tmpls
}

csa-nevt(){ echo ${CSA_NEVT:-1} ; }

csa-envsetup(){

   if [ -f "$(csa-envcache)" ]; then 
       csa-envcache-source
   else
       csa-nuwaenv
       $ENV_HOME/base/envcap/envcap.py -zpost -tcsa save cachediff
   fi

   # non-cached envvars
   csa-export   
}


csa-flags-notes(){ cat << EON

G4CERENKOV_COLLECT_STEP
G4SCINTILLATION_COLLECT_STEP
       collect G4 gensteps and send to chroma, currently with noreturn

G4CERENKOV_COLLECT_PHOTON
G4SCINTILLATION_COLLECT_PHOTON
       collect G4 generated photons and send to chroma,
       allows same steps comparison with the chroma generated photons  

G4SCINTILLATION_COLLECT_PROP
       collect scintillation integrals into G4DAEPropList 
       and write to file, for debugging  

EON
}

csa-flags-(){ cat << EOF
G4SCINTILLATION_COLLECT_STEP
G4SCINTILLATION_COLLECT_PHOTON
G4SCINTILLATION_COLLECT_PROP
G4CERENKOV_COLLECT_STEP
G4CERENKOV_COLLECT_PHOTON
EOF
}
csa-flags(){ $FUNCNAME- | tr "\n" "," ; }

csa-modulo-photon(){ echo 1 ; }   # 1 for all photons, eg 100 for big scaledown

csa-nuwarun(){
   local msg=" === $FUNCNAME :"

   csa-envsetup

   local args="DetSimChroma.csa --modulo-photon $(csa-modulo-photon) --use-basic-physics --chroma --chroma-flags $(csa-flags) --test $*"

   echo $msg $args 

   nuwa.py -n $(csa-nevt) -m "$args"
}


csa-nuwarun-pid(){ echo $(pgrep -f nuwa.py) ; }
csa-dbg(){  csa-nuwarun-gdb ; }
csa-nuwarun-gdb(){
   
   local def=$(csa-nuwarun-pid)
   local pid=${1:-$def}
   [ -z $pid ] && echo enter pid of nuwa.py process && return 1

   csa-envsetup

   gdb $(which python) $pid
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
