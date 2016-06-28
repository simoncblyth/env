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


issue : not producing exports on G5 
-------------------------------------------

Seems to run but no files::

    [blyth@ntugrid5 ~]$ export.sh VGDX DayaBay
    === dyb-- : defining dyb function
    === dyb-- : invoking "dyb dybgaudi"
    #CMT> The tag ntu_tarurl is not used in any tag expression. Please check spelling
    === dyb-- : invoking "dyb dybdbi"
    #CMT> The tag ntu_tarurl is not used in any tag expression. Please check spelling
    G4DAE_EXPORT_LOG=/home/blyth/local/env/geant4/geometry/export/DayaBay_VGDX_20160519-1934/export.log
    G4DAE_EXPORT_SITE=DayaBay
    G4DAE_EXPORT_DIR=/home/blyth/local/env/geant4/geometry/export/DayaBay_VGDX_20160519-1934
    G4DAE_EXPORT_SEQUENCE=VGDX

On D GDML symbols in libG4persistency.dylib::

   simon:lib blyth$ nm libG4persistency.dylib | grep GDML | c++filt

Huh on G5 no such lib, need DYBX perhaps::

    blyth@ntugrid5 Linux-g++]$ l libG4pers*
    ls: cannot access libG4pers*: No such file or directory
    [blyth@ntugrid5 Linux-g++]$ pwd
    /home/blyth/local/env/dyb/external/build/LCG/geant4.9.2.p01/lib/Linux-g++



Relation of export-name to GCache idp dir 
-------------------------------------------

GCache dir is beneath the export-name dir as it represents, via the digest 
in the name, the geometry volume selection used to create the cache. 

::

    delta:cu blyth$ export-;export-name dyb
    /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    delta:cu blyth$ idp
    delta:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ 
    delta:g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae blyth$ pwd
    /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae


How the export works
---------------------

Critical part of nuwa.py module export_all.py hooks up the export G4 RunAction::

     69     # --- WRL + GDML + DAE geometry export ---------------------------------
     70     from GaussTools.GaussToolsConf import GiGaRunActionExport, GiGaRunActionCommand, GiGaRunActionSequence
     71     export = GiGaRunActionExport("GiGa.GiGaRunActionExport")
     72     
     73     giga.RunAction = export
     74     giga.VisManager = "GiGaVisManager/GiGaVis"
     75     
     76     import DetSim 
     77     DetSim.Configure(physlist=DetSim.physics_list_basic,site=site)

::

    delta:env blyth$ find . -name 'GiGaRunActionExport.*'
    ./geant4/geometry/GaussTools/src/Components/GiGaRunActionExport.cpp
    ./geant4/geometry/GaussTools/src/Components/GiGaRunActionExport.h

::

    530 void GiGaRunActionExport::WriteDAE(G4VPhysicalVolume* wpv, const G4String& path, G4bool recreatePoly  )
    531 {
    532 #ifdef EXPORT_G4DAE
    533    if(path.length() == 0 || wpv == 0){
    534        std::cout << "GiGaRunActionExport::WriteDAE invalid path OR NULL PV  " << path << std::endl ;
    535        return ;
    536    }
    537    std::cout << "GiGaRunActionExport::WriteDAE to " << path << " recreatePoly " << recreatePoly << std::endl ;
    538    G4DAEParser parser ;
    539    G4bool refs = true ;
    540    G4int nodeIndex = -1 ;   // so World is volume 0 
    541    parser.Write(path, wpv, refs, recreatePoly, nodeIndex );
    542 #else
    543    std::cout << "GiGaRunActionExport::WriteDAE BUT this installation  not compiled with -DEXPORT_G4DAE " << std::endl ;
    544 #endif
    545 }




Explore Id Mapping export
--------------------------

::

   export.sh MX DayaBay

Notable Exports
-------------------

To make exports easily accessible from multiple machines, 
they are stored on cms02 webserver. Grab the g4_00.dae 
from the export directory on N with::

    [root@cms02 downloads]# export-pull Lingao_VGDX_20140414-1247


Belle7 down, so custom copy
----------------------------

::

    cmd="scp N:$(NODE_TAG=N EXPORT_EXT=gdml export-name dyb) $(EXPORT_EXT=gdml export-name dyb)"
    echo $cmd
       scp N:/data1/env/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml
    eval $cmd


Getting Exports
-----------------

Browse available exports at 

* http://dayabay.phys.ntu.edu.tw/env/geant4/geometry/export/

Grab with::

   export-
   export-get 


Getting CerenkovStep/ScintillationStep files
----------------------------------------------

::

    delta:~ blyth$ export-cerenkov-get
    mkdir -p /usr/local/env/cerenkov && scp G5:/home/blyth/local/env/cerenkov/1.npy /usr/local/env/cerenkov/1.npy

    delta:~ blyth$ export-cerenkov-get | sh 
    1.npy                                                                                               100%  747KB 746.9KB/s   00:01    
     

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
export-home(){ echo $(local-base)/env/geant4/geometry/export ; }
export-sdir(){ echo $ENV_HOME/geant4/geometry/export ; }
export-cd(){  cd $(export-home); }
export-scd(){  cd $(export-sdir); }
export-mate(){ mate $(export-dir) ; }
export-url(){ echo http://dayabay.phys.ntu.edu.tw/env/geant4/geometry/export ; }

export-edit(){
   local name=${1:-dyb}
   local path=$(export-name $name)
   echo $msg name $name path $path : MANUAL EDITS WOULD BE UNWISE
   vi $path
}

export-ext(){ echo ${EXPORT_EXT:-dae} ; }
export-name(){ echo $(export-base $1).$(export-ext) ; }
export-base(){
  local base=$(export-home)
  case $1 in 
       dyb) echo $base/DayaBay_VGDX_20140414-1300/g4_00 ;;
       dybf) echo $base/DayaBay_VGDX_20140414-1300/g4_00 ;;
       dpib) echo $base/dpib/cfg4 ;; 
       far) echo $base/Far_VGDX_20140414-1256/g4_00 ;;
    lingao) echo $base/Lingao_VGDX_20140414-1247/g4_00 ;;
       lxe) echo $base/LXe/g4_00 ;;
       jpmt) echo $base/juno/test3 ;;
       juno) echo $base/juno/nopmt ;;
       jtst) echo $base/juno/test ;;
  esac
}

export-strip-extra-meta(){

   local orig=$1
   local nometa=${orig/.dae/.nometa.dae}
   echo $msg orig $orig nometa $nometa

   xsltproc $(export-home)/strip-extra-meta.xsl $orig > $nometa
}


export-geometry(){
  case $1 in 
        dyb) echo 3153:12221 ;;   # skip RPC and radslabs  
        dybf) echo 2+,3147+ ;;   
       juno) echo 1:25000  ;;
       jpmt) echo 1:25000  ;;
       jtst) echo 1:25000  ;;
        lxe) echo 1:   ;;
       dpib) echo 1:   ;;
  esac
}


export-export(){
   export DAE_NAME=$(export-name dyb)
   export DAE_NAME_DYB=$(export-name dyb)
   export DAE_NAME_DPIB=$(export-name dpib)

   export DAE_NAME_DYB_GDML=$(EXPORT_EXT=gdml export-name dyb)
   export DAE_NAME_DYB_GDML_FAKESD="/dd/Geometry/PMT/lvPmtHemiCathode;/dd/Geometry/PMT/lvHeadonPmtCathode"

   ## NB names from multiple exports : 
   ##      once stabilised the names should all have a common export base
   ##
   export DAE_NAME_DYB_IDMAP=/usr/local/env/geant4/geometry/export/DayaBay_MX_20141013-1711/g4_00.idmap   
   #export DAE_NAME_DYB_TRANSFORMCACHE=/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml.cache
   export DAE_NAME_DYB_TRANSFORMCACHE=/usr/local/env/geant4/geometry/export/DybG4DAEGeometry.cache
   export DAE_NAME_DYB_CHROMACACHE=$LOCAL_BASE/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae.29c299d81706c62884caf5c3dbdea5c1/chroma_geometry
   export DAE_NAME_DYB_CHROMACACHE_MESH=$DAE_NAME_DYB_CHROMACACHE/chroma.detector:Detector:0x10a881cd0/mesh/chroma.geometry:Mesh:0x10ea3bf50
   export DAE_NAME_DYB_NOEXTRA=$(EXPORT_EXT=dae.noextra.dae export-name dyb)
   export DAE_NAME_DYBF=$(export-name dybf)
   export DAE_NAME_FAR=$(export-name far)
   export DAE_NAME_LIN=$(export-name lingao)
   export DAE_NAME_LXE=$(export-name lxe)

   export DAE_NAME_JUNO=$(export-name juno)
   export DAE_NAME_JPMT=$(export-name jpmt)
   export DAE_NAME_JTST=$(export-name jtst)

   export DAE_GEOMETRY_DYB=$(export-geometry dyb)
   export DAE_GEOMETRY_DYBF=$(export-geometry dybf)
   export DAE_GEOMETRY_DPIB=$(export-geometry dpib)
   export DAE_GEOMETRY_JUNO=$(export-geometry juno)
   export DAE_GEOMETRY_JPMT=$(export-geometry jpmt)
   export DAE_GEOMETRY_JTST=$(export-geometry jtst)
   export DAE_GEOMETRY_LXE=$(export-geometry lxe)

   $FUNCNAME-templates
}

export-path-template(){ 
   case $1 in
           photon) echo "$(local-base)/env/tmp/%s.npy" ;;
       #     foton) echo "$(local-base)/env/tmp/%s.npy" ;;
                *) echo "$(local-base)/env/$1/%s.npy" ;;
   esac
}

export-source-node(){ echo G5 ; }
export-photon-get(){           export-npy-get ${1:-1} photon ; }

export-gopscintillation-get(){ export-npy-get ${1:-1} gopscintillation ; }
export-gopcerenkov-get(){      export-npy-get ${1:-1} gopcerenkov ; }

export-opscintillation-get(){  export-npy-get ${1:-1} opscintillation ; }
export-opcerenkov-get(){       export-npy-get ${1:-1} opcerenkov ; }

export-scintillation-get(){    export-npy-get ${1:-1} scintillation ; }
export-cerenkov-get(){         export-npy-get ${1:-1} cerenkov ; }


export-prop-rget(){            export-npy-rget prop ; }


export-type-notes(){ cat << EON

photon
   Source unspecified optical photons, possibly hits 


gopscintillation
   G4 generated scintilation photon

gopcerenkov
   G4 generated cerenkov photon


opscintillation
   Chroma generated scintilation photon

opcerenkov
   Chroma generated cerenkov photon


oxscintillation
   OptiX generated scintilation photon

oxcerenkov
   OptiX generated cerenkov photon




scintillation
   Scintillation Generation Step

cerenkov
   Cerenkov Generation Step


EON
}

export-type-(){ cat << EOT
photon
gopscintillation
gopcerenkov
opscintillation
opcerenkov
scintillation
cerenkov
EOT
}


export-du(){ export-op "du -h" ${1:-1} ; }
export-ls(){ export-op "ls -l" ${1:-1} ; }
export-op(){ 
   local op=${1:-ls}
   local evt=${2:-1}
   local type
   local path
   export-type- | while read type ; do  
      path=$(export-npy-path $evt $type)
      cmd="$op $path"
      echo $cmd
      eval $cmd
   done 
}

export-steps-get(){  
    export-scintillation-get ${1:-1}
    export-cerenkov-get ${1:-1}
}

export-gop-get(){  
    export-gopscintillation-get ${1:-1}
    export-gopcerenkov-get ${1:-1}
}



export-npy-path(){
    local evt=${1:-1}
    local typ=${2:-cerenkov}
    local tmpl=$(export-path-template $typ)
    local path=$(printf $tmpl $evt)
    echo $path
}
export-npy-rget(){
    local evt="dummy"
    local typ=${1:-prop}
    local from=$(export-source-node)
    [ "$NODE_TAG" == "$from" ] && echo $msg cannot get to self && return 

    local rpath=$(NODE_TAG=$from export-npy-path $evt $typ)
    local lpath=$(export-npy-path $evt $typ)
    local rdir=$(dirname $rpath)
    local ldir=$(dirname $lpath)

    local cmd="mkdir -p $ldir && scp -r $from:$rdir/\*.npy $ldir/" 
    echo $cmd

}

export-npy-get(){

    local evt=${1:-1}
    local typ=${2:-cerenkov}

    local from=$(export-source-node)
    [ "$NODE_TAG" == "$from" ] && echo $msg cannot get to self && return 

    local rpath=$(NODE_TAG=$from export-npy-path $evt $typ)
    local lpath=$(export-npy-path $evt $typ)

    local cmd="mkdir -p $(dirname $lpath) && scp $from:$rpath $lpath" 
    echo $cmd
}


export-export-templates(){
   export DAE_PATH_TEMPLATE_ROOT="$LOCAL_BASE/env/tmp/%s.root"
   export DAE_PATH_TEMPLATE_NPY="$LOCAL_BASE/env/tmp/%s.npy"
   export DAE_PATH_TEMPLATE=$DAE_PATH_TEMPLATE_NPY

   #export DAE_HIT_PATH_TEMPLATE=$(export-path-template hit)
   #export DAE_PHOTON_PATH_TEMPLATE=$(export-path-template photon)

   #export DAE_GOPSCINTILLATION_PATH_TEMPLATE=$(export-path-template gopscintillation)
   #export DAE_GOPCERENKOV_PATH_TEMPLATE=$(export-path-template gopcerenkov)

   #export DAE_OPCERENKOV_PATH_TEMPLATE=$(export-path-template opcerenkov)
   #export DAE_OPSCINTILLATION_PATH_TEMPLATE=$(export-path-template opscintillation)

   #export DAE_OXCERENKOV_DESCRIPTION="Cerenkov photons generated from OptiX based machinery"
   #export DAE_OXCERENKOV_PATH_TEMPLATE=$(export-path-template oxcerenkov)

   #export DAE_OXSCINTILLATION_DESCRIPTION="Scintillation photons generated from OptiX based machinery"
   #export DAE_OXSCINTILLATION_PATH_TEMPLATE=$(export-path-template oxscintillation)

   #export DAE_RXCERENKOV_DESCRIPTION="Cerenkov records generated from OptiX based machinery"
   #export DAE_RXCERENKOV_PATH_TEMPLATE=$(export-path-template rxcerenkov)


   #export DAE_RXSCINTILLATION_DESCRIPTION="Scintillation records generated from OptiX based machinery"
   #export DAE_RXSCINTILLATION_PATH_TEMPLATE=$(export-path-template rxscintillation)

   #export DAE_OPCERENKOVGEN_PATH_TEMPLATE=$(export-path-template opcerenkovgen)
   #export DAE_OPSCINTILLATIONGEN_PATH_TEMPLATE=$(export-path-template opscintillationgen)

   #export DAE_CERENKOV_PATH_TEMPLATE=$(export-path-template cerenkov)
   #export DAE_SCINTILLATION_PATH_TEMPLATE=$(export-path-template scintillation)

   #export DAE_TEST_PATH_TEMPLATE=$(export-path-template test)
   #export DAE_PROP_PATH_TEMPLATE=$(export-path-template prop)

   #export DAE_PMTHIT_PATH_TEMPLATE=$(export-path-template pmthit)
   #export DAE_G4PMTHIT_PATH_TEMPLATE=$(export-path-template g4pmthit)

   #export DAE_CHROMAPHOTON_PATH_TEMPLATE=$(export-path-template chromaphoton)


   #export DAE_DOMAIN_DESCRIPTION="Domain float quad parameters used to decode packed data such as photon records"
   #export DAE_DOMAIN_PATH_TEMPLATE=$(export-path-template domain)

   #export DAE_IDOMAIN_DESCRIPTION="Domain integer quad parameters eg max_bounce maxrec "
   #export DAE_IDOMAIN_PATH_TEMPLATE=$(export-path-template idomain)

   #export DAE_SEQCERENKOV_DESCRIPTION="Photon sequence index "
   #export DAE_SEQCERENKOV_PATH_TEMPLATE=$(export-path-template seqcerenkov)

   #export DAE_SEQSCINTILLATION_DESCRIPTION="Photon sequence index "
   #export DAE_SEQSCINTILLATION_PATH_TEMPLATE=$(export-path-template seqscintillation)



}

export-lambda-(){ cat << EOL

manually add the below to ipython environment using ipython-edit

EOL
}



export-juno-get(){
   cd $(export-home)
   scp lxslc5:~lint/test.dae juno/$1.dae
}



export-get(){
   local tag=${1:-DayaBay_VGDX_20140414-1300}
   local tagdir=$(export-home)/$tag
   local name=g4_00.$(export-ext)
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

export-nuwapkg(){ echo $DYB/NuWa-trunk/lhcb/Sim/GaussTools/src/Components ; }
export-nuwapkg-cd(){  cd $(export-nuwapkg) ; }

export-prep(){
   local arg=${1:-MX}
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
   export-scd

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


export-copy-()
{
   local ext=${1:-dae}
   local dest=$HOME/opticksdata/export
   local base=$(export-home)
   local path
   local dpath
   local rel
   local cmd
   find $base -name "g4_00.$ext" | while read path 
   do
      rel=${path/$base\/}
      dpath=$dest/$rel 
      ddir=$(dirname $dpath)
      [ ! -d "$ddir" ] && echo mkdir -p $ddir
      [ ! -f "$dpath" ] && echo cp $path $dpath
   done   


}  


export-copy-all-()
{
   export-copy- dae
   export-copy- gdml
   export-copy- idmap
}  

export-copy-detector-dir-()
{
   local dirname=${2:-GPmt}
   local detector=${2:-DayaBay}
   local base=$(export-home)
   local dest=$HOME/opticksdata/export
   local path
   local ddir
   find $base -type d -name $dirname | while read path 
   do
       ddir=$dest/$detector/$dirname 
       dfold=$(dirname $ddir)
       [ ! -d "$dfold" ] && echo mkdir -p $dfold
       [ ! -d "$ddir" ] && echo cp -r $path $dfold/
   done



}




#export-main $*


