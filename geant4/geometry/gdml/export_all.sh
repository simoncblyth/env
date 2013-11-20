#!/bin/bash -l

export_dir(){
   local pfx=${1}
   local pwd=$(pwd -P)
   local rdir=${pwd/$ENV_HOME\/}
   local tag=$(date +"%Y%m%d-%H%M")
   local xdir=$LOCAL_BASE/env/$rdir/${pfx}_${tag}
   mkdir -p $xdir
   echo $xdir
}
export_banner(){
   echo 
   echo ========== $* ==================
   echo 
}

export_run(){
   local arg=${1:-VDG}
   export G4DAE_EXPORT_SEQUENCE="$arg"
   export G4DAE_EXPORT_EXIT="1"  
   export G4DAE_EXPORT_DIR=$(export_dir $G4DAE_EXPORT_SEQUENCE)

   nuwa.py -G $XMLDETDESCROOT/DDDB/dayabay.xml -n1 -m export_all

   export_banner G4DAE
   env | grep G4DAE

   export_banner $G4DAE_EXPORT_DIR
   ls -l $G4DAE_EXPORT_DIR
}

export_main(){
   fenv
   cd $ENV_HOME/geant4/geometry/gdml
   export_run VDGVDG
   export_run DVGDVG
}

export_main

