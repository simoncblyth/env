# === func-gen- : geant4/geometry/export/export fgp geant4/geometry/export/export.bash fgn export fgh geant4/geometry/export
export-src(){      echo geant4/geometry/export/export.bash ; }
export-source(){   echo ${BASH_SOURCE:-$(env-home)/$(export-src)} ; }
export-vi(){       vi $(export-source) ; }
export-env(){      elocal- ; }
export-usage(){ cat << EOU

EXPORT GEANT4 GEOMETRY INTO VRML, GDML AND DAE 
================================================

From script usage::

   export.sh VGD 
   export.sh DGV
   export.sh VGD gdb


Notable Exports
-------------------

* First Far site export.

  * N:/data1/env/local/env/geant4/geometry/export/VGDX_20140414-1149/g4_00.dae
  * http://dayabay.phys.ntu.edu.tw/downloads/VGDX_20140414-1149-g4_00.dae

Export Environment Setup
--------------------------

Following nuwa integration changes need to use DYBX installation "geant4_with_dae" for this
otherwise the export runs without error but no exports happen, as preprocessor not enabled in the 
compile of standard installation

 
ACTION CONTROLLED BY ENVVARS
------------------------------

**G4DAE_EXPORT_SEQUENCE**
     envvar is set based on the script argument
     controls the formats and their order of export

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


BACKGROUND
-------------

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/Visualization/visexecutable.html


FUNCTIONS
-----------

**export-cf**
     parses WRL and DAE files writing SQLite DB with geom, point and face tables. 
     Connects the DB into an sqlite3 session. 


MALLOC DEBUGGING
-----------------

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
   shift
   local site=${1:-DayaBay}
   case $site in 
     DayaBay|Dayabay|dayabay|dyb) site=DayaBay ;;
                   Lingao|lingao) site=Lingao ;; 
                         Far|far) site=Far ;; 
   esac
   # controlling argument quoting is tedious, so slip in the arguments via envvar 
   export G4DAE_EXPORT_SITE="$site"
   export G4DAE_EXPORT_SEQUENCE="$arg"
   export G4DAE_EXPORT_DIR=$(export-dir ${G4DAE_EXPORT_SITE}_${G4DAE_EXPORT_SEQUENCE})
   export G4DAE_EXPORT_LOG=$G4DAE_EXPORT_DIR/export.log
   env | grep G4DAE
}


export-pull(){
   [ "$NODE_TAG" != "C2R" ] && echo $msg RUN THIS ON WEBSERVER AS C2R && return

   local name=${1:-DayaBay_VGDX_20140414-1149}
   apache-
   cd $(apache-htdocs)/downloads

   mkdir -p $name
   scp N:/data1/env/local/env/geant4/geometry/export/$name/g4_00.dae $name/g4_00.dae
}


export-run(){

   export-prep $*
   local log=$G4DAE_EXPORT_LOG
   export-banner $msg writing nuwa.py output to $log

   #LIBC_FATAL_STDERR_=1 MALLOC_CHECK_=1 nuwa.py $(export-args $*)  > $log 2>&1 

   nuwa.py $(export-args)  > $log 2>&1
     

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

export-setup(){
  #
   # fenv   nope these are from the standard installation
   #
   dyb-- dybdbi   # after setting DYB to DYBX, some random pkg 
}


export-main(){
   export-setup
   export-cd

   if [ -z "$GDB" ]; then 
      export-run $*
   else
      export-gdb $*
   fi

   export-post
}

export-cf(){
   [ -n "$G4DAE_EXPORT_DIR" ] && cd $G4DAE_EXPORT_DIR
   pwd
   type $FUNCNAME

   local base=g4_00
   local dae=$base.dae
   local wrl=$base.wrl
   local sql=$base.sql

   [ ! -f "$dae.db" ] && daedb.py --daepath $dae
   [ ! -f "$wrl.db" ] && vrml2file.py --save $wrl
   [ ! -f "$sql" ] && export-cf-sql- $base > $sql

   cat $sql
   which sqlite3
   sqlite3 -version
   sqlite3 -init $sql

}

export-cf-sql-(){ 
   local base=$1
   cat << EOU
-- $FUNCNAME $base
attach database "$base.dae.db" as dae ;
attach database "$base.wrl.db" as wrl ;
.databases
.mode column
.header on 

-- sqlite3 -init $base.sql

EOU
}

export-cf-find-(){
  local def="g4_00.dae.db"
  local ptn="${1:-$def}"
  local i=0
  find . -name "$ptn" | while read line ; do 
    i=$(( $i + 1 ))
cat << EOL
attach database "$line" as db$i ; 
EOL
  done

cat << EOT
.database
.mode column
.header on
EOT

}



#export-main $*


