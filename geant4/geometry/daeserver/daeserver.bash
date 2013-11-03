# === func-gen- : geant4/geometry/daeserver/daeserver fgp geant4/geometry/daeserver/daeserver.bash fgn daeserver fgh geant4/geometry/daeserver
daeserver-src(){      echo geant4/geometry/daeserver/daeserver.bash ; }
daeserver-source(){   echo ${BASH_SOURCE:-$(env-home)/$(daeserver-src)} ; }
daeserver-vi(){       vi $(daeserver-source) ; }
daeserver-env(){      elocal- ; }
daeserver-usage(){ cat << EOU

DAESERVER
=========




EOU
}
daeserver-dir(){ echo $(env-home)/geant4/geometry/daeserver ; }
daeserver-cd(){  cd $(daeserver-dir); }
daeserver-mate(){ mate $(daeserver-dir) ; }
daeserver-get(){
   local dir=$(dirname $(daeserver-dir)) &&  mkdir -p $dir && cd $dir

}
