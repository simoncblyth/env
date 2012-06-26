alias p-vgm="scp $HOME/.bash_vgm P:"
[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/vgm.bash


vgm-usage(){ cat << EOU


EOU
}

##
## VGM Virtual Geometry Model ...
##     provides conversions between:
##        root TGeo, 
##        Geant4, 
##        GDML and AGDD
##
##   usage:
##       vgm-get 
##       vgm-make
##
##       vgm-test-make
##       vgm-test-run
##
##   issues...
##      1) the .gdml created (on linux x86_64) has spaces between attribute label and value
##         ... root TGeoManager::Import doesnt accept this
##
##      2) the .gdml created has very odd names with _constA_constA_ etcc many
##          many times  .. esp for logicMovalbeTF
##
##
##      2)  TGeoManager::Import( "Aberdeen_World_attcot.gdml" )  results in
##           many ... 
##         Error: Unsupported GDML Tag Used :rotationref. Please Check Geometry/Schema.
##      TGDMLParse::ParseGDML which is used by TGeoManager
##    gets confused somehow...
##       is looking for "position" and "rotation" ... not "positionref" and
##      
##   VGM     :
##              depends on standard things only   IMaterial , IBox ....
##            (interfaces only)
##
##             VGM/common/Math.h 
##                  defines M_PI and includes <math.h>
##                  potential Darwin header search issue, as case insensitive ????
##
##          
##   BaseVGM  :   VGM
##                                            VMaterial
##   RootGM : VGM BaseVGM and root TGeo classes
##
##   Geant4GM : VGM BaseVGM and G4Material, G4Element,....
##
##   XmlVGM :   VGM BaseVGM ClhepVGM  
##           cannot find xercesc dependency
##

#export VGM_NAME=vgm.2.08.03     ## ivana tested with  CLHEP 2.0.2.3, Geant4 8.1, Root 5.13/04 
export VGM_NAME=vgm.2.08.04      ## ivana tested with  CLHEP 2.0.3.1, Geant4 8.2, Root 5.14/00 
export VGM_INSTALL=$LOCAL_BASE/vgm/$VGM_NAME

if [ "$G4SYSTEM" == "Darwin-g++" ]; then
  ## the G4SYSTEM of Darwin-g++ doesnt work 
  export VGM_SYSTEM=Darwin
## for the VGM examples
  export DYLD_LIBRARY_PATH=$VGM_INSTALL/lib/$VGM_SYSTEM:$DYLD_LIBRARY_PATH
else
  export VGM_SYSTEM=$G4SYSTEM	
  export LD_LIBRARY_PATH=$VGM_INSTALL/lib/$VGM_SYSTEM:$LD_LIBRARY_PATH
fi

export VGM_CMT="VGM_prefix:$VGM_INSTALL VGM_system:$VGM_SYSTEM"
export ENV2GUI_VARLIST="VGM_INSTALL:VGM_SYSTEM:$ENV2GUI_VARLIST"

## undocumented required environment

if [ "X$ROOTSYS" != "X" ]; then
  export ROOT_LIBDIR=$ROOTSYS/lib
fi

if [ "X$G4LIB" != "X" ]; then
   export G4_LIBDIR=$G4LIB/$G4SYSTEM
fi

#export DYLD_LIBRARY_PATH=$VGM_INSTALL/lib/$VGM_SYSTEM:$DYLD_LIBRARY_PATH


vgm-get(){

  n=$VGM_NAME
  cd $LOCAL_BASE
  test -d vgm || ( $SUDO mkdir vgm && $SUDO chown $USER vgm )
  cd vgm
  
  tgz=$n.tar.gz
  test -f $tgz || curl -o $tgz http://ivana.home.cern.ch/ivana/$tgz 
  test -d $n  || tar zxvf $tgz 

}


vgm-wipe(){
  cd $VGM_INSTALL
  
  echo wiping vgm distro beneath $VGM_INSTALL back to pristine state
  test -d $VGM_INSTALL && cd $VGM_INSTALL && rm -rf tmp/$VGM_SYSTEM && rm -rf lib/$VGM_SYSTEM

}

vgm-diff(){
  
   cd $LOCAL_BASE/vgm
   diff -r --brief pristine/$VGM_NAME $VGM_NAME
}


vgm-switch-math(){
   cd $VGM_INSTALL

   flist="./packages/XmlVGM/source/Maps.cxx ./packages/Geant4GM/source/solids/Para.cxx ./packages/Geant4GM/source/solids/Sphere.cxx ./packages/BaseVGM/source/solids/VPolyhedra.cxx"
   for fl in $flist
   do
      # cat $fl
	  # grep "Math.h" $fl 
      perl -pi -e 's/Math.h/VMath.h/' $fl
   done
   mv ./packages/VGM/include/VGM/common/Math.h ./packages/VGM/include/VGM/common/VMath.h

}



vgm-crash(){
   cd $VGM_INSTALL/test
   . env_setup_darwin.sh
   echo enter args : VGM Root Geant4 AGDD Solids debug noVis  
   gdb /usr/local/vgm/vgm.2.08.04/test/bin/Darwin-g++/vgm_test

}

vgm-test-make(){

  cd $VGM_INSTALL/test

  ## NB the setup sets G4WORKDIR that is critical to G4 flavored builds
  ##

  
  if [ "$VGM_SYSTEM" == "Darwin" ]; then
	  
     cp -f env_setup.sh env_setup_darwin.sh
     perl -pi -e 's/LD_LIBRARY_PATH/DYLD_LIBRARY_PATH/g' env_setup_darwin.sh
     . env_setup_darwin.sh
	 dlp
  else
     . env_setup.sh 
  fi

  if [ "$LOCAL_NODE" == "g4pb" ]; then
     make OGLLIBS="$OGLLIBS -L/usr/local/openmotif/lib" VGM_DEBUG=1 G4DEBUG=1 CPPVERBOSE=1
  else
     make VGM_DEBUG=1 G4DEBUG=1 CPPVERBOSE=1
  fi

}








vgm-test-run(){

  cd $VGM_INSTALL/test
  #./test1_suite.sh
  #./test2_suite.sh

  exe=$VGM_INSTALL/test/bin/$G4SYSTEM/vgm_test
   
  if [ "$LOCAL_NODE" == "g4pb" ]; then
      dlp
      otool -L $exe
  fi
  
  ls -alst $exe 
   
  echo enter : r VGM Root Geant4 AGDD Solids debug noVis  
  $exe  VGM Root Geant4 AGDD Solids debug noVis
  
}


vgm-correct(){

  if  [ "$VGM_SYSTEM" == "Darwin" ]; then

      if ([ "$VGM_NAME" == "vgm.2.08.04" ] || [ "$VGM_NAME" == "vgm.2.08.03" ]) ; then

         echo applying a correction as seem to be missing a few libraries...
		 echo this is validated by comparison with the cmt/requirements file

         cd $VGM_INSTALL/config
	     rm -f geant4.gmk.original 
	     perl -pi.original -e 's/(-lG4geometry -lG4Materials)\s*$/$1  -lG4graphics_reps -lG4intercoms -lG4global/'  geant4.gmk
         diff geant4.gmk.original geant4.gmk

      else
		  echo .bash_vgm caution not applying a correction used in another verison 
	  fi


  else
	  echo correctiono onlty needed for Darwin nor $VGM_SYSTEM
  fi

}


vgm-make(){

  cd $VGM_INSTALL/packages
  if [ "$VGM_SYSTEM" == "Darwin" ]; then
	 make   OGLLIBS="$OGLLIBS -L/usr/local/openmotif/lib" VGM_DEBUG=1 CPPVERBOSE=1
  else
     make
  fi


  ##  caution VGM Geant4GM requires :
  ##     1) global and dynamic Geant4 libs
  ##     2) the Geant4 includes in $G4INSTALL/include
  ##
  ##
  ##  missing libG4geometry ...   presumably need to rebuild G4 without granular ?
  ##   actually "granular" is default have to pick "global" libraries 
  ##
  ##  missing path for OpenGL
  ##    -L/lib -lGLU -lGL -lXm -lXpm 
  ##  fixed by setting OGLLIBS 
  ##   
  ##  and motif ???
  ##  my XMLIBS seems to be ignored by the VGM build ?
  ##    -L/usr/local/openmotif/lib -lXm -lXpm
  ##
  ##  looks like version mismatch between VGM and G4 ???
  ##
  ##  [g4pb:/usr/local/vgm/vgm.2.08.03/packages] blyth$ make LOAD_XM=1
  ##
  ##  still ignoring my XMLIBS
  ##   [g4pb:/usr/local/vgm/vgm.2.08.03/config] blyth$ vi sys/Darwin.gmk
  ##   ... seems hardcoded in above so set it appropriately
  ##
  ##
  ##  Creating global shared library
  ##  /usr/local/vgm/vgm.2.08.03/lib/Darwin/libGeant4GM.dylib
  ##  libdir=`(cd /usr/local/vgm/vgm.2.08.03/lib/Darwin;/bin/pwd)`; cd
  ##  /usr/local/vgm/vgm.2.08.03/tmp/Darwin; g++ -dynamiclib -twolevel_namespace
  ##  -undefined error -dynamic -single_module -o $libdir/libGeant4GM.dylib
  ##  Geant4GM_materials/*.o Geant4GM_solids/*.o Geant4GM_volumes/*.o
  ##  -L/usr/local/clhep/clhep-1.9.2.3-osx104_ppc_gcc401/lib -lCLHEP
  ##  -L/usr/local/vgm/vgm.2.08.03/lib/Darwin -lClhepVGM -lBaseVGM
  ##  -L/usr/local/geant4/dist/2/geant4.8.1.p01/lib/Darwin-g++ -lG4geometry
  ##  -lG4Materials -L/usr/local/lib -lSoXt -lCoin -L/usr/X11R6/lib -lGLU -lGL
  ##  -L/usr/local/openmotif/lib -lXm -lXpm -L/usr/X11R6/lib  -lXmu -lXt -lXext
  ##  -lX11 -lSM -lICE;
  ##  ld: Undefined symbols:
  ##  _G4cerr
  ##  _G4cout
  ##  __Z11G4ExceptionPKc
  ##  __ZN10G4BestUnitC1EdRK8G4String
  ##  __ZN10G4BestUnitD1Ev
  ##  __ZlsRSo10G4BestUnit
  ##  __ZN13G4UIdirectoryC1EPKc
  ##  __ZN13G4UImessenger15GetCurrentValueEP11G4UIcommand
  ##  __ZN13G4UImessengerC2Ev
  ##  __ZN13G4UImessengerD2Ev
  ##  __ZN18G4UIcmdWithAString13SetCandidatesEPKc
  ##
  ##
  ##
  ## libdir=`(cd /usr/local/vgm/vgm.2.08.03/lib/Darwin;/bin/pwd)`; cd
  ##  ##/usr/local/vgm/vgm.2.08.03/tmp/Darwin; g++ -dynamiclib -twolevel_namespace
  ## -undefined error -dynamic -single_module -o $libdir/libGeant4GM.dylib
  ## Geant4GM_materials/*.o Geant4GM_solids/*.o Geant4GM_volumes/*.o
  ## -L/usr/local/clhep/clhep-1.9.2.3/lib -lCLHEP
  ## -L/usr/local/vgm/vgm.2.08.03/lib/Darwin -lClhepVGM -lBaseVGM
  ## -L/usr/local/geant4/geant4.8.1.p01/lib/Darwin-g++ -lG4geometry -lG4Materials
  ## -L/usr/local/soxt/lib -lSoXt -L/usr/local/coin3d/lib -lCoin -L/usr/X11R6/lib
  ## -lGLU -lGL -lXm -lXpm -L/usr/X11R6/lib  -lXmu -lXt -lXext -lX11 -lSM -lICE;
  ## /usr/bin/libtool: can't locate file for: -lXm
  ## /usr/bin/libtool: file: -lXm is not an object file (not allowed in a library)
  ## make[1]: *** [/usr/local/vgm/vgm.2.08.03/lib/Darwin/libGeant4GM.dylib] Error 1
  ## Making dependency for file transform.cxx ...
  ## Making dependency for file axis.cxx ...
  ##
  ##
  ##
  ## 
  ##   the G4 switches seem to be ignored for Darwin ????
  ##    OGLLIBS setting works though
  ## [g4pb:/usr/local/vgm/vgm.2.08.03/packages] blyth$ make G4UI_USE=1 G4VIS_USE=1  OGLLIBS="$OGLLIBS -L/usr/local/openmotif/lib" 
  ##
  ##     
  ##   the setting of XMLIBS here was ignored...
  ## 
  ##  [g4pb:/usr/local/vgm] blyth$ diff vgm.2.08.03/config/sys/Darwin.gmk vgm.2.08.03.keep/config/sys/Darwin.gmk 
  ## <   XMLIBS    := -lXm -lXpm
  ##---
  ## > # SCB ... correct the XMLIBS
  ## > #  XMLIBS    := -lXm -lXpm
  ## >   XMLIBS    := -L/usr/local/openmotif/lib -lXm -lXpm
  ##
  ##
  ##   Mar20 
  ##
#Creating global shared library
#/usr/local/vgm/vgm.2.08.04/lib/Darwin/libGeant4GM.dylib
#libdir=`(cd /usr/local/vgm/vgm.2.08.04/lib/Darwin;/bin/pwd)`; cd
#/usr/local/vgm/vgm.2.08.04/tmp/Darwin; g++ -dynamiclib -twolevel_namespace
#-undefined error -dynamic -single_module -o $libdir/libGeant4GM.dylib
#Geant4GM_materials/*.o Geant4GM_solids/*.o Geant4GM_volumes/*.o
#-L/usr/local/clhep/clhep-1.9.2.3/lib -lCLHEP
#-L/usr/local/vgm/vgm.2.08.04/lib/Darwin -lClhepVGM -lBaseVGM
#-L/usr/local/geant4/geant4.8.1.p01/lib/Darwin-g++ -lG4geometry -lG4Materials
#-L/usr/local/soxt/lib -lSoXt -L/usr/local/coin3d/lib -lCoin -L/usr/X11R6/lib
#-lGLU -lGL -lXm -lXpm -L/usr/X11R6/lib  -lXmu -lXt -lXext -lX11 -lSM -lICE;
#/usr/bin/libtool: can't locate file for: -lXm
#/usr/bin/libtool: file: -lXm is not an object file (not allowed in a library)
#make[1]: *** [/usr/local/vgm/vgm.2.08.04/lib/Darwin/libGeant4GM.dylib] Error 1

 }
