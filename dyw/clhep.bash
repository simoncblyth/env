alias p-clhep="scp $HOME/.bash_clhep P:"

##
##  NB compiling clhep is compulsory for G4dyb , as CMT gets the CLHEP info
##      from the clhep-config script , 
##
##   Usage 
##           G> p-clhep
##
##           P> ini  
##           P> clhep-get
##           P> clhep-configure
##           P> clhep-build
##
## version 1.9.2.3 recommended for Geant4 8.1
## version 2.0.3.1 recommended for Geant4 8.2
## NB changing the clhep version means recompiling VGM, G4, ... etc
##
##
#CLHEP_VERS="1.9.2.3"   
CLHEP_VERS="2.0.3.1" 
CLHEP_NAME=clhep-$CLHEP_VERS

CLHEP_FOLDER=$LOCAL_BASE/clhep/$CLHEP_NAME
CLHEP_BASE_DIR=$CLHEP_FOLDER
CLHEP_CMT="CLHEP_config:$CLHEP_FOLDER/bin/clhep-config"
ENV2GUI_VARLIST="CLHEP_BASE_DIR:$ENV2GUI_VARLIST"

if [ "$CMTCONFIG" == "Darwin" ]; then
	export DYLD_LIBRARY_PATH=$CLHEP_FOLDER/lib:$DYLD_LIBRARY_PATH 
else
    export   LD_LIBRARY_PATH=$CLHEP_FOLDER/lib:$LD_LIBRARY_PATH
fi

export ENV2GUI_VARLIST
export CLHEP_CMT
export PATH=$CLHEP_BASE_DIR/bin:$PATH

## this is consumed by Geant4  ... /usr/local/clhep/clhep-1.9.2.3
export CLHEP_BASE_DIR



clhep-get(){

  n=$CLHEP_NAME
  cd $LOCAL_BASE
  test -d clhep || ( $SUDO mkdir clhep && $SUDO chown $USER clhep )
  cd clhep
  mkdir -p $n $n-build $n-src
  cd $n-src
  tgz=$n.tgz 
  url=http://proj-clhep.web.cern.ch/proj-clhep/DISTRIBUTION/distributions/$tgz

  echo getting $url to $tgz 
  curl -o $tgz $url && ( tar zxvf $tgz || echo error maybe url changed )
  

}

clhep-configure(){

   [ "$CMTCONFIG" == "Darwin" ] && echo MACOSX_DEPLOYMENT_TARGET $MACOSX_DEPLOYMENT_TARGET
   n=$CLHEP_NAME
   v=$CLHEP_VERS

   cd $LOCAL_BASE/clhep/$n-build
    ## ../$n-src/$v/CLHEP/configure --prefix=$LOCAL_BASE/clhep/$n --enable-exceptions CXXFLAGS="-g"
    ../$n-src/$v/CLHEP/configure --prefix=$LOCAL_BASE/clhep/$n --enable-exceptions 

}


clhep-build(){

   [ "$CMTCONFIG" == "Darwin" ] && echo MACOSX_DEPLOYMENT_TARGET $MACOSX_DEPLOYMENT_TARGET
   n=$CLHEP_NAME
   cd $LOCAL_BASE/clhep/$n-build
   make clean 
   make 
   make check 
   make install

   make docs
 #  make install-docs
 # ERROR FROM install-docs
 #
 # cd doc; make  install-docs
 # make[2]: Entering directory `/usr/local/clhep-build/Exceptions/doc'
 # /usr/local/clhep-src/1.9.3.1/CLHEP/Exceptions/autotools/install-sh -d
 # /usr/local/clhep/doc/Exceptions
 # /usr/bin/install -c -m 644 ../doc/HepTuple-exceptions
 # /usr/local/clhep/doc/Exceptions/HepTuple-exceptions
 # /usr/bin/install: cannot stat `../doc/HepTuple-exceptions': No such file or
 # directory
 # make[2]: *** [install-docs] Error 1
 #
  
}


