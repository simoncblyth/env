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

export_usage(){ cat << EOU

Meaning of the G4DAE_EXPORT_SEQUENCE control characters:

V
   VRML WriteVis, same as "IF"
I
   VRML InitVis  
F
   VRML FlushVis  

D
   Write DAE
G
   Write GDML
C
   Clean SolidStore
X
   Abrupt Exit

EOU
}


export_cd(){ cd $(export_home) ; }
export_home(){ echo $ENV_HOME/geant4/geometry/gdml ; }
export_args(){ cat << EOA
     -G $XMLDETDESCROOT/DDDB/dayabay.xml -n1 -m export_all
EOA
}

export_run(){
   local arg=${1:-VDG}
   export G4DAE_EXPORT_SEQUENCE="$arg"
   export G4DAE_EXPORT_DIR=$(export_dir $G4DAE_EXPORT_SEQUENCE)
   local log=$G4DAE_EXPORT_DIR/export_all.log
   export_banner $msg writing nuwa.py output to $log
   nuwa.py $(export_args)  > $log 2>&1
   export_banner $msg wrote nuwa.py output to $log
   export_banner G4DAE
   env | grep G4DAE
   export_banner $G4DAE_EXPORT_DIR
   ls -l $G4DAE_EXPORT_DIR
}

export_cf(){
   export_run VDGVDGX
   #export_run DVGDVGX

   export_post
}

export_dbg(){
   #export G4DAE_EXPORT_SEQUENCE="VDGX"
   export G4DAE_EXPORT_SEQUENCE="DVGX"
   export G4DAE_EXPORT_DIR=$(export_dir $G4DAE_EXPORT_SEQUENCE)
   env | grep G4DAE
   local cmd="gdb $(which python) --args $(which python) $(which nuwa.py) $(export_args)"
   echo $cmd
   eval $cmd 

   export_post
}


export_post(){
   cd $G4DAE_EXPORT_DIR
   pwd
}

export_main(){
   local mode=$1
   fenv
   export_cd
   case $mode in 
       dbg) export_dbg ;;
         *) export_cf  ;;
   esac
}

export_main $*

