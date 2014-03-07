# === func-gen- : geant4/geometry/dae fgp geant4/geometry/dae.bash fgn dae fgh geant4/geometry
dae-src(){      echo geant4/geometry/dae.bash ; }
dae-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dae-src)} ; }
dae-vi(){       vi $(dae-source) ; }
dae-env(){      elocal- ; }
dae-usage(){ cat << EOU

DAE based on GDML code
========================

* http://www.khronos.org/files/collada_spec_1_4.pdf 
* https://github.com/KhronosGroup/COLLADA-CTS/blob/master/StandardDataSets/collada/library_nodes/node/1inst2inLN/1inst2inLN.dae

  * a good source of non-trivial collada examples 

The *rotate* element contains a list of four floating-point values, 
similar to rotations in the OpenGL and RenderMan specification. 
These values are organized into a column vector [X,Y,Z] 
specifying the axis of rotation followed by an angle in degrees. 

Schema Validation
------------------

* attribute 'id': '/dd/Materials/PPE0x92996b8' is not a valid value of the atomic type 'xs:ID'
* attribute 'sid': '/dd/Materials/PPE0x92996b8' is not a valid value of the atomic type 'xs:NCName'

Curiously get 'xs:ID' validation errors on G, but not N 

* http://www.schemacentral.com/sc/xsd/t-xsd_ID.html

  * The type xsd:ID is used for an attribute that uniquely identifies an element in an XML document. 

Strip Extras
-----------------

The extra nodes need to be stripped to avoid schema validation errors (why are extra content not ignored ?)::

    dae-strip-extra g4_00.dae.6 > g4_00.dae.6.noextra
    dae-validate  g4_00.dae.6.noextra



Code Organisation
--------------------

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

Maybe can restructure making G4DAE* chain of classes to inherit 
from the G4GDML* ones. This would avoid duplication and allow
G4DAE to incorporate GDML into extra tags.


FUNCTIONS
----------

dae-get
        grab latest .dae from N via http


LCG Builder-ization
---------------------

Ape the geant4 build with ::

    fenv
    cd $SITEROOT/lcgcmt/LCG_Builders/geant4/cmt
    cmt config
    . setup.sh

    cmt pkg_get      # looks to not use the script
    cmt pkg_config
    cmt pkg_make
    cmt pkg_install

Considered a separate LCG_Builder for g4dae, but that 
makes things more complicated. It is essentially a
rather extended patch against Geant4.

See also geant4/geometry/export/nuwa_integration



EOU
}
dae-dir(){ echo $(env-home)/geant4/geometry/DAE ; }
dae-cd(){  cd $(dae-dir); }
dae-mate(){ mate $(dae-dir) ; }


dae-strip-extra(){
  xsltproc $(env-home)/geant4/geometry/collada/strip-extra.xsl $1 
}

dae-install(){
   dae-install-lib
   dae-install-inc
}

dae-install-inc(){
   nuwa-
   local blib=$(env-home)/geant4/geometry/DAE/include
   local ilib=$(nuwa-g4-idir)/include
   ls -l $blib
   #ls -l $ilib
   local cmd="cp $blib/G4DAE*.{hh,icc} $ilib/"
   echo "$cmd"
   eval $cmd
   ls -l $ilib/G4DAE*
}

dae-install-lib(){
   nuwa-
   local name=libG4DAE.so
   local blib=$(nuwa-g4-bdir)/lib/Linux-g++/$name
   local ilib=$(nuwa-g4-idir)/lib/$name
   local cmd="cp $blib $ilib"
   echo $cmd
   eval $cmd
   ls -l $blib $ilib
}


dae-make(){
   local tgt=$1
   nuwa-
   make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_DAE=1 G4LIB_USE_DAE=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) G4INSTALL=$(nuwa-g4-bdir) CPPVERBOSE=1  $tgt
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

dae-validate(){
   local pth=${1:-$(dae-pth)}
   xmllint --noout --schema $(dae-xsd) $pth
}

dae-name(){ echo g4_01.dae ; }
dae-xsd(){  echo $(env-home)/geant4/geometry/DAE/schema/collada_schema_1_4.xsd  ; }
dae-url(){  echo http://belle7.nuu.edu.tw/dae/$(dae-name) ; }
dae-pth(){  echo $LOCAL_BASE/env/geant4/geometry/xdae/$(dae-name) ; }
dae-edit(){ vi $(dae-pth) ; }
dae-get(){
   local url=$(dae-url)
   local pth=$(dae-pth)
   local nam=$(basename $url)
   local cmd="curl -o $pth $url "
   echo $cmd
   eval $cmd

   dae-info
}

dae-info(){
   local pth=$(dae-pth)
   ls -l $pth
   du -h $pth
   wc -l $pth
}

