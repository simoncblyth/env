# === func-gen- : geant4/g4ten fgp geant4/g4ten.bash fgn g4ten fgh geant4
g4ten-src(){      echo geant4/g4ten.bash ; }
g4ten-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4ten-src)} ; }
g4ten-vi(){       vi $(g4ten-source) ; }
g4ten-env(){      elocal- ; g4ten-export ; }
g4ten-usage(){ cat << EOU

Geant4 Ten
===========

See also *g4beta-*

INSTALLS 
----------



EOU
}
g4ten-dir(){ echo $(local-base)/env/geant4/$(g4ten-name) ; }
g4ten-bdir(){ echo $(g4ten-dir).build ; }
g4ten-idir(){ echo $(g4ten-dir).local ; }
g4ten-cd(){  cd $(g4ten-dir); }
g4ten-bcd(){  cd $(g4ten-bdir); }
g4ten-name(){ echo geant4.10.00 ; }
g4ten-url(){ echo http://geant4.cern.ch/support/source/$(g4ten-name).tar.gz ; }
g4ten-get(){
   local dir=$(dirname $(g4ten-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4ten-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz

   mkdir -p $(g4ten-bdir)
}
g4ten-gdml(){ echo $(g4ten-dir)/source/persistency/gdml ; }
g4ten-export(){
  export G4TEN_GDML=$(g4ten-gdml) 
}


