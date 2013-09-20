# === func-gen- : geant4/geometry/export/shapedb fgp geant4/geometry/export/shapedb.bash fgn shapedb fgh geant4/geometry/export
shapedb-src(){      echo geant4/geometry/vrml2/shapedb.bash ; }
shapedb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(shapedb-src)} ; }
shapedb-vi(){       vi $(shapedb-source) ; }
shapedb-env(){      elocal- ; }
shapedb-usage(){ cat << EOU

SHAPEDB
=======



EOU
}

shapedb-sdir(){ echo $(env-home)/geant4/geometry/vrml2 ; }
shapedb-scd(){  cd $(shapedb-sdir); }

shapedb-dir(){  echo $(local-base)/env/geant4/geometry/vrml2 ; }
shapedb-cd(){  cd $(shapedb-dir); }

shapedb-path(){ echo $(shapedb-dir)/$(shapedb-name) ; }
shapedb-name(){ echo g4_01.db ; }
shapedb-sh(){ sqlite3 $(shapedb-path) ; }
