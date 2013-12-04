# === func-gen- : geant4/geometry/export/export fgp geant4/geometry/export/export.bash fgn export fgh geant4/geometry/export
export-src(){      echo geant4/geometry/export/export.bash ; }
export-source(){   echo ${BASH_SOURCE:-$(env-home)/$(export-src)} ; }
export-vi(){       vi $(export-source) ; }
export-env(){      elocal- ; }
export-usage(){ cat << EOU

From script usage::

   export.sh VD gdb



Meaning of the G4DAE_EXPORT_SEQUENCE control characters:

V
   VRML WriteVis, same as "IF"
I
   VRML InitVis  
F
   VRML FlushVis  
D
   Write DAE, without poly recreation : unless not existing already 
A
   Write DAE, with forced poly recreation
G
   Write GDML
C
   Clean SolidStore
X
   Abrupt Exit



MALLOC_CHECK_=1 
     Propagated to libc M_CHECK_ACTION, 1 means deatailed error message but continue

LIBC_FATAL_STDERR_=1 
     Rather than /dev/tty to allow redirection, to allow correlation of the corruption with stdout 


EOU
}
export-dir(){ 
   local pfx=${1}
   local pwd=$(pwd -P)
   local rdir=${pwd/$ENV_HOME\/}
   local tag=$(date +"%Y%m%d-%H%M")
   local xdir=$LOCAL_BASE/env/$rdir/${pfx}_${tag}
   mkdir -p $xdir
   echo $xdir
}
export-home(){ echo $(env-home)/geant4/geometry/export ; }
export-cd(){  cd $(export-home); }
export-mate(){ mate $(export-dir) ; }
export-get(){
   local dir=$(dirname $(export-dir)) &&  mkdir -p $dir && cd $dir

}


export-grep(){ pgrep -f $(export-module) ; }
export-kill(){ pkill -f $(export-module) ; }

export-banner(){
   echo 
   echo ========== $* ==================
   echo 
}

export-module(){ echo export_all ; }
export-args(){ cat << EOA
     -G $XMLDETDESCROOT/DDDB/dayabay.xml -n1 -m $(export-module)
EOA
}


export-prep(){
   local arg=${1:-VX}
   export G4DAE_EXPORT_SEQUENCE="$arg"
   export G4DAE_EXPORT_DIR=$(export-dir $G4DAE_EXPORT_SEQUENCE)
   export G4DAE_EXPORT_LOG=$G4DAE_EXPORT_DIR/export.log
   env | grep G4DAE
}

export-run(){
   export-prep $*
   local log=$G4DAE_EXPORT_LOG
   export-banner $msg writing nuwa.py output to $log
   LIBC_FATAL_STDERR_=1 MALLOC_CHECK_=1 nuwa.py $(export-args)  > $log 2>&1

   export-banner $msg wrote nuwa.py output to $log
   export-banner G4DAE
   env | grep G4DAE
   export-banner $G4DAE_EXPORT_DIR
   ls -l $G4DAE_EXPORT_DIR
}

export-gdb(){
   export-prep $*
   local cmd="gdb $(which python) --args $(which python) $(which nuwa.py) $(export-args)"
   echo $cmd
   eval $cmd 
}

export-post(){
   cd $G4DAE_EXPORT_DIR
   pwd
}

export-main(){
   local arg=$1
   shift
   local cmd=$1
   fenv
   export-cd
   case $cmd in 
       gdb) export-gdb $arg ;;
         *) export-run $arg  ;;
   esac
   export-post
}

#export-main $*


