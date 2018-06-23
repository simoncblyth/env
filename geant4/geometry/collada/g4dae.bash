# === func-gen- : geant4/geometry/collada/g4dae fgp geant4/geometry/collada/g4dae.bash fgn g4dae fgh geant4/geometry/collada
eg4dae-src(){      echo geant4/geometry/collada/g4dae.bash ; }
eg4dae-source(){   echo ${BASH_SOURCE:-$(env-home)/$(eg4dae-src)} ; }
eg4dae-vi(){       vi $(eg4dae-source) ; }
eg4dae-env(){      elocal- ; }
eg4dae-usage(){ cat << EOU

G4DAE
======

FUNCTIONS
----------

*eg4dae-cf*
       open sqlite3 session with dae and wrl databases attached

EOU
}
eg4dae-dir(){ echo $(env-home)/geant4/geometry/collada ; }
eg4dae-ddir(){ echo $(local-base)/env/geant4/geometry/collada ; }
eg4dae-cd(){  cd $(eg4dae-dir)/$1 ; }
eg4dae-mate(){ mate $(eg4dae-dir) ; }
eg4dae-get(){
   local dir=$(dirname $(eg4dae-dir)) &&  mkdir -p $dir && cd $dir
}

eg4dae-daedbpath(){
  case ${1:-1} in
    0) echo "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.db"     ;;
    1) echo "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae.db" ;;
   10) echo "$LOCAL_BASE/env/geant4/geometry/gdml/g4_10.dae.db" ;;
  esac
}
eg4dae-wrldbpath(){
  case ${1:-0} in
    0) echo "$LOCAL_BASE/env/geant4/geometry/vrml2/g4_01.db"  ;;
  esac
}




eg4dae-prep(){
   if [ "$NODE_TAG" == "N" ]; then 
      python-
      python- source
   fi 
   vrml2file.py --save --noshape g4_00.wrl 
   daedb.py --daepath g4_00.dae
}


eg4dae-cf-(){  cat << EOS
attach database "g4_00.dae.db" as dae ;
attach database "g4_00.wrl.db" as wrl ;
.databases
.mode column
.header on 

-- sqlite3 -init cf.sql
--
EOS
}
eg4dae-cf(){
   local sql=cf.sql
   $FUNCNAME- $* > $sql
   sqlite3-
   case $NODE_TAG in 
      N) sqlite3-- -init $sql ;;    # avoid glacial system sqlite3 on N
      *) sqlite3   -init $sql ;;
   esac
}
