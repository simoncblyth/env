# === func-gen- : geant4/geometry/collada/g4dae fgp geant4/geometry/collada/g4dae.bash fgn g4dae fgh geant4/geometry/collada
g4dae-src(){      echo geant4/geometry/collada/g4dae.bash ; }
g4dae-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4dae-src)} ; }
g4dae-vi(){       vi $(g4dae-source) ; }
g4dae-env(){      elocal- ; }
g4dae-usage(){ cat << EOU

G4DAE
======

FUNCTIONS
----------

*g4dae-cf*
       open sqlite3 session with dae and wrl databases attached

EOU
}
g4dae-dir(){ echo $(env-home)/geant4/geometry/collada ; }
g4dae-ddir(){ echo $(local-base)/env/geant4/geometry/collada ; }
g4dae-cd(){  cd $(g4dae-dir)/$1 ; }
g4dae-mate(){ mate $(g4dae-dir) ; }
g4dae-get(){
   local dir=$(dirname $(g4dae-dir)) &&  mkdir -p $dir && cd $dir
}

g4dae-daedbpath(){
  case ${1:-1} in
    0) echo "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.db"     ;;
    1) echo "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae.db" ;;
   10) echo "$LOCAL_BASE/env/geant4/geometry/gdml/g4_10.dae.db" ;;
  esac
}
g4dae-wrldbpath(){
  case ${1:-0} in
    0) echo "$LOCAL_BASE/env/geant4/geometry/vrml2/g4_01.db"  ;;
  esac
}


g4dae-cf-(){  cat << EOS
attach database "g4_00.dae.db" as dae ;
attach database "g4_00.wrl.db" as wrl ;
.databases
.mode column
.header on 

-- sqlite3 -init cf.sql
--
EOS
}
g4dae-cf(){
   local sql=cf.sql
   $FUNCNAME- $* > $sql
   sqlite3-
   case $NODE_TAG in 
      N) sqlite3-- -init $sql ;;    # avoid glacial system sqlite3 on N
      *) sqlite3   -init $sql ;;
   esac
}
