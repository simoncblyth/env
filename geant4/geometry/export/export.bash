# === func-gen- : geant4/geometry/export/export fgp geant4/geometry/export/export.bash fgn export fgh geant4/geometry/export
export-src(){      echo geant4/geometry/export/export.bash ; }
export-source(){   echo ${BASH_SOURCE:-$(env-home)/$(export-src)} ; }
export-vi(){       vi $(export-source) ; }
export-env(){      elocal- ; }
export-usage(){ cat << EOU

EXPORT GEANT4 GEOMETRY INTO VRML, GDML AND DAE 
================================================

From script usage::

   export.sh VGDX DayaBay
   export.sh VGDX Lingao
   export.sh VGDX Far

   GDB=1 export.sh VGDX Far


Notable Exports
-------------------

To make exports easily accessible from multiple machines, 
they are stored on cms02 webserver. Grab the g4_00.dae 
from the export directory on N with::

    [root@cms02 downloads]# export-pull Lingao_VGDX_20140414-1247


Getting Exports
-----------------

Browse available exports at 

* http://dayabay.phys.ntu.edu.tw/env/geant4/geometry/export/

Grab with::

   export-
   export-get 


Administrator: Placing Exports
-----------------------------------

Formerly kept in cms02 apache htdocs::

    [blyth@cms02 env]$ ll /data/env/system/apache/httpd-2.0.64/htdocs/env/geant4/geometry/export/
    total 40
    drwxr-xr-x  2 blyth blyth 4096 Apr 14 12:51 Lingao_VGDX_20140414-1247
    drwxr-xr-x  2 blyth blyth 4096 Apr 14 12:59 Far_VGDX_20140414-1256
    drwxr-xr-x  2 blyth blyth 4096 Apr 14 13:04 DayaBay_VGDX_20140414-1300

    [blyth@cms02 export]$ du -hs * 
    6.9M    DayaBay_VGDX_20140414-1300
    8.6M    Far_VGDX_20140414-1256
    6.9M    Lingao_VGDX_20140414-1247


#. env-htdocs-rsync assuming creation into htdocs. Export machinery predates that.
#. new approach, need to push to bitbucket from /Users/blyth/simoncblyth.bitbucket.org/env/geant4/geometry/export/
   in order to appear http://simoncblyth.bitbucket.org/env/geant4/geometry/

   * hmm, bitbucket does not provide automatic directories, 
 
     * generate some html OR base on RST and include in notes ?


::

    delta:geometry blyth$ pwd
    /Users/blyth/simoncblyth.bitbucket.org/env/geant4/geometry
    delta:geometry blyth$ l
    total 0
    drwxr-xr-x  4 blyth  staff  136 Apr 27 11:57 collada

For now, just manually grab::

   delta:export blyth$ scp -r N:/data1/env/local/env/geant4/geometry/export/DayaBay_MX_20140916-2050 .


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
export-home(){ echo $LOCAL_BASE/env/geant4/geometry/export ; }
export-cd(){  cd $(export-home); }
export-mate(){ mate $(export-dir) ; }
export-url(){ echo http://dayabay.phys.ntu.edu.tw/env/geant4/geometry/export ; }



export-name(){
  local base=$(export-home)
  case $1 in 
       dyb) echo $base/DayaBay_VGDX_20140414-1300/g4_00.dae ;;
       dybf) echo $base/DayaBay_VGDX_20140414-1300/g4_00.dae ;;
       far) echo $base/Far_VGDX_20140414-1256/g4_00.dae ;;
    lingao) echo $base/Lingao_VGDX_20140414-1247/g4_00.dae ;;
       lxe) echo $base/LXe/g4_00.dae ;;
       juno) echo $base/juno/test2.dae ;;
  esac
}

export-geometry(){
  case $1 in 
        dyb) echo 3153:12221 ;;   # skip RPC and radslabs  
        dybf) echo 2+,3147+ ;;   
       juno) echo 1:25000  ;;
        lxe) echo 1:   ;;
  esac
}


export-export(){
   export DAE_NAME=$(export-name dyb)
   export DAE_NAME_DYB=$(export-name dyb)
   export DAE_NAME_DYBF=$(export-name dybf)
   export DAE_NAME_FAR=$(export-name far)
   export DAE_NAME_LIN=$(export-name lingao)
   export DAE_NAME_LXE=$(export-name lxe)
   export DAE_NAME_JUNO=$(export-name juno)
   export DAE_PATH_TEMPLATE="/usr/local/env/tmp/%(arg)s.root"


   export DAE_GEOMETRY_DYB=$(export-geometry dyb)
   export DAE_GEOMETRY_DYBF=$(export-geometry dybf)
   export DAE_GEOMETRY_JUNO=$(export-geometry juno)
   export DAE_GEOMETRY_LXE=$(export-geometry lxe)

}

export-juno-get(){
   cd $(export-home)
   scp lxslc5:~lint/test.dae juno/$1.dae
}




export-get(){
   local tag=${1:-DayaBay_VGDX_20140414-1300}
   local tagdir=$(export-home)/$tag
   local name=g4_00.dae
   local url=$(export-url)/$tag/$name

   [ ! -d "$tagdir" ] && mkdir -p $tagdir
   [ -f "$tagdir/$name" ] && echo $url already downloaded to $tagdir  && return 

   local cmd="curl $url -o $tagdir/$name"
   echo $msg $cmd
   eval $cmd

   ls -l $tagdir
}

export-names-(){
  curl -s $(export-url)/ | perl -n -e 'm,href="([-\w]*)/", && print "$1\n"' -
}

export-get-all(){
  local name
  export-names- | while read name ; do
     echo name $name
     export-get $name
  done
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


