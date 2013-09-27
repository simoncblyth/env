# === func-gen- : geant4/geometry/gdml/gdmldb fgp geant4/geometry/gdml/gdmldb.bash fgn gdmldb fgh geant4/geometry/gdml
gdmldb-src(){      echo geant4/geometry/gdml/gdmldb.bash ; }
gdmldb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gdmldb-src)} ; }
gdmldb-vi(){       vi $(gdmldb-source) ; }
gdmldb-env(){      elocal- ; }
gdmldb-usage(){ cat << EOU





EOU
}
gdmldb-dir(){ echo $(local-base)/env/geant4/geometry/gdml/geant4/geometry/gdml-gdmldb ; }
gdmldb-cd(){  cd $(gdmldb-dir); }
gdmldb-mate(){ mate $(gdmldb-dir) ; }
gdmldb-get(){
   local dir=$(dirname $(gdmldb-dir)) &&  mkdir -p $dir && cd $dir

}

gdmldb-path(){ echo $(local-base)/env/geant4/geometry/gdml/g4_01.gdml.db ; }
gdmldb-sh(){ sqlite3 $(gdmldb-path) ; }
