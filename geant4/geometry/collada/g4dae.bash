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

g4dae-cf-path(){ echo  $(g4dae-ddir)/g4dae-cf.sql ; }
g4dae-cf-(){  cat << EOS
attach database "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.db" as dae ;
attach database "$LOCAL_BASE/env/geant4/geometry/vrml2/g4_01.db" as wrl ;
.databases
EOS
}
g4dae-cf(){
   local sql=$(g4dae-cf-path)
   mkdir -p $(dirname $sql)
   $FUNCNAME- > $sql
   sqlite3 -init $sql
}
