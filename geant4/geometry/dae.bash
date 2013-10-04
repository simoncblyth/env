# === func-gen- : geant4/geometry/dae fgp geant4/geometry/dae.bash fgn dae fgh geant4/geometry
dae-src(){      echo geant4/geometry/dae.bash ; }
dae-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dae-src)} ; }
dae-vi(){       vi $(dae-source) ; }
dae-env(){      elocal- ; }
dae-usage(){ cat << EOU

DAE based on GDML code
========================

GDML inheritance heirarchy::

     G4GDMLWriteStructure
     G4GDMLWriteParamvol 
     G4GDMLWriteSetup
     G4GDMLWriteSolids
     G4GDMLWriteMaterials 
     G4GDMLWriteDefine        
     G4GDMLWrite              

DAE inheritance heirarchy::

     G4DAEWriteStructure   library_nodes         (based on G4GDMLWriteStructure)
     G4DAEWriteParamvol
     G4DAEWriteSetup       library_visual_scenes (based on G4GDMLWriteSetup)
     G4DAEWriteSolids      library_geometries    (based on G4GDMLWriteSolids) 
     G4DAEWriteMaterials   library_materials     (based on G4GDMLWriteMaterials) 
     G4DAEWriteEffects     library_effects       (originated) 
     G4DAEWriteAsset       asset                 (based on G4GDMLWriteDefine)             
     G4DAEWrite            COLLADA    



EOU
}
dae-dir(){ echo $(env-home)/geant4/geometry/DAE ; }
dae-cd(){  cd $(dae-dir); }
dae-mate(){ mate $(dae-dir) ; }


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



