# === func-gen- : geant4/geometry/dae fgp geant4/geometry/dae.bash fgn dae fgh geant4/geometry
dae-src(){      echo geant4/geometry/dae.bash ; }
dae-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dae-src)} ; }
dae-vi(){       vi $(dae-source) ; }
dae-env(){      elocal- ; }
dae-usage(){ cat << EOU

DAE based on GDML code
========================

::




EOU
}
dae-dir(){ echo $(local-base)/env/geant4/geometry/geant4/geometry-dae ; }
dae-cd(){  cd $(dae-dir); }
dae-mate(){ mate $(dae-dir) ; }
dae-get(){
   local dir=$(dirname $(dae-dir)) &&  mkdir -p $dir && cd $dir

}


dae-install(){
   nuwa-
   local name=libG4DAE.so
   local blib=$(nuwa-g4-bdir)/lib/Linux-g++/$name
   local ilib=$(nuwa-g4-idir)/lib/$name
   local cmd="cp $blib $ilib"
   echo $cmd
   eval $cmd
   ls -l $blib $ilib
}


dae-switch(){
   perl -pi -e 's,GDML,DAE,g' *.*
}

dae-mv(){
   local name
   local newname
   local cmd
   ls -1 G4GDML*.* | while read name ; do
      newname=${name/GDML/DAE}
      cmd="mv $name $newname"
      echo $cmd
   done
}



