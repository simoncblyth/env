



#========================== GDML

## not GDML_INC as documentation suggests
export GDML_BASE="/usr/local/gdml" 
export GDML_LIB="${GDML_BASE}/lib" 
export GDML_INC="${GDML_BASE}/include" 
#export DYLD_LIBRARY_PATH=${GDML_LIB}:${DYLD_LIBRARY_PATH}

#export GDML_HOME="${GDML_BASE}/GDML_2_8_0"
export GDML_HOME="${GDML_BASE}/cvs/GDML2"
export GDML_XSD=$GDML_HOME/GDMLSchema/gdml.xsd

#export DYLD_LIBRARY_PATH=${GDML_LIB}/dylib:$DYLD_LIBRARY_PATH 


gdml-build(){

  #  initially fails because $G4INCLUDE directory is not filled with headers
  #  so return to geant4...
  #    cd $GQ_HOME/source ; make includes
  #
  #  had to add a const to get to compile:
  #  /usr/local/gdml/GDML_2_8_0/CPPGDML/G4Binding/G4Subscribers/src/materialSubscriber.cpp
  #
  #  thence run into -soname lack ... so "make clean" and  flip to static
  #
  cd $GDML_HOME/CPPGDML 
  ./configure \
              --prefix=/usr/local/gdml \
              --with-platform=${G4SYSTEM} \
    		  --enable-shared-libs=no \
			  --enable-static-libs=yes \
		      --enable-geant4-granular-libs=yes \
			  --enable-gdml-verbose=yes \
		      --enable-compile-verbose=no 

   cat config/make/local_settings.gmk

   make 
   make install

   # NB the install overwrote the priot /usr/local/gdml/{lib,bin,include} with
   # the cvs version 

}

_gdml-example(){

	cd $HOME/geant4/$GQ_NAME/examples/extended/gdml 
    gmake 


## had to edit the GNUmakefile for the headers to be found
#
#  CPPFLAGS    += \
#        -I$(GDML_BASE)/include \
# -I$(GDML_BASE)/include/Common/Saxana \
# -I$(GDML_BASE)/include/Common/Writer \
#		-I$(GDML_BASE)/include/Common/Schema \
#		-I$(GDML_BASE)/include/G4Binding/G4Writer \
#		-I$(GDML_BASE)/include/G4Binding/G4Processor \
#		-I$(GDML_BASE)/include/G4Binding/G4Evaluator \
#		-I$(XERCESCROOT)/include
#																	
#



   perl -pi -e "s|/afs/cern.ch/sw/lcg/app/releases/GDML/pro/schema/gdml_2.0.xsd|$GDML_XSD|" test.gdml
  
}

gdml-example(){
	cd $HOME/geant4/$GQ_NAME/examples/extended/gdml 
	exe=../../../bin/Darwin-g++/g4_gdml_read_write
	ls -aslt $exe
	$exe
	ls -aslt $exe
}


gdml-package-setup(){   ## interface GDML to G4dyb with CMT
 
    n=GDML
	v=v0

    cd $DYW/External
	cmt remove $n $v
	cmt create $n $v

}


gdml-get(){
  cd /usr/local/gdml/cvs
  ##
  ## cvs -d :pserver:anonymous@simu.cvs.cern.ch:/cvs/simu co -r HEAD -d . GDML2
  ## this fails with : cvs checkout: existing repository /cvs/simu does not match /cvs/simu/simu/GDML2
  ##
  ## so try :
  ## cvs -d :pserver:anonymous@simu.cvs.cern.ch:/cvs/simu/simu co -r HEAD -d . GDML2
  ## fails with : cvs [checkout aborted]: unrecognized auth response from simu.cvs.cern.ch: cvs [pserver aborted]: /cvs/simu/simu: no such repository
  ##
  ## so try :
  ## cvs -d :pserver:anonymous@simu.cvs.cern.ch:/cvs/simu co -r HEAD -d . simu/GDML2
  ##
  ## finally works (Jan 4th 2007):
  cvs -d :pserver:anonymous@simu.cvs.cern.ch:/cvs/simu co  GDML2

  ##
  ## cvs co -H        for help on "co"
  ##        -d dir    checks out into specified directory  (rather than that of the module name) 
  ##        -r rev    selects a tagged revision 
  ##
}




#========================== GDML examples ($GDML_HOME/CPPGDML/Examples )

##alias g4gogdml="cd $GDML_HOME/CPPGDML/Examples/g4gogdml ; /usr/local/gdml/bin/g4gogdml_static " 
alias _g4gogdml="cd $HOME/geant4/$GQ_NAME/examples/extended/g4gogdml" ;
alias g4gogdml="cd $HOME/geant4/$GQ_NAME/examples/extended/g4gogdml ; ../../bin/$G4SYSTEM/g4gogdml " 

#
# copied in the cvs updates to this example:
# cp gdmlExamples/g4gogdml/split.gdml g4gogdml/
# cp gdmlExamples/g4gogdml/materials.xml g4gogdml/
# cp gdmlExamples/g4gogdml/auxiliary.gdml g4gogdml/
# cp gdmlExamples/g4gogdml/README    g4gogdml/
#
#  the Jan4/2007 cvs GDML build succeeds to make .dylib , BUT
#  want to force them not to be used, as my geant4 currently doesnt have .dylib 
#  so tuck them away, and dont expose GDML_LIB to DYLD
#
#cd $HOME/geant4/examples/g4gogdml


#========================== GDML usage (in geant4/examples/extended/gdml)





