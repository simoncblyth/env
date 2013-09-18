# === func-gen- : geant4/geometry/export/shapedb fgp geant4/geometry/export/shapedb.bash fgn shapedb fgh geant4/geometry/export
shapedb-src(){      echo geant4/geometry/export/shapedb.bash ; }
shapedb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(shapedb-src)} ; }
shapedb-vi(){       vi $(shapedb-source) ; }
shapedb-env(){      elocal- ; }
shapedb-usage(){ cat << EOU

SHAPEDB
=======



EOU
}
shapedb-dir(){ echo $(env-home)/geant4/geometry/export ; }
shapedb-cd(){  cd $(shapedb-dir); }
shapedb-mate(){ mate $(shapedb-dir) ; }
shapedb-sh(){
   shapedb-cd
   sqlite3 g4_01.db
}
