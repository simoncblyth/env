#!/bin/bash -l

export_dir(){
   local pwd=$(pwd -P)
   local rdir=${pwd/$ENV_HOME\/}
   local tag=$(date +"%Y%m%d-%H%M")
   local xdir=$LOCAL_BASE/env/$rdir/$tag
   mkdir -p $xdir
   echo $xdir
}

fenv
cd $ENV_HOME/geant4/geometry/gdml

export G4DAE_EXPORT_DIR=$(export_dir)
export G4DAE_EXPORT_SEQUENCE="DVVVDDDGGGVVVVDVDVD"
export G4DAE_EXPORT_EXIT="1"  

nuwa.py -G $XMLDETDESCROOT/DDDB/dayabay.xml -n1 -m export_all



