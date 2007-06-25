
geant4-use-x(){ scp $HOME/$DYW_BASE/geant4_use.bash ${1:-$TARGET_TAG}:$DYW_BASE; }

#  
#   Testing geant4 recipe
#      1)  geant4-copy-examples
#      2)  _a01      build an example
#      3)  a01       run the example
#

###### G4 environment ##################

if [ -x "${GQ_HOME}/env.sh" ]; then

      ## keep the workdir versioned
      ## export G4WORKDIR=$HOME/geant4/$GQ_NAME 
      ##	
      ## hmm is this needed at runtime ???
	  if ([ "$NODE_TAG" == "G1" ] || [ "$NODE_TAG" == "P" ] || [ "$NODE_TAG" == "$CLUSTER_TAG" ] ); then  ## grid1 suffers from non-sync times so everything must reside on the same disk
         #export G4WORKDIR=$GQ_HOME/workdir
         export G4WORKDIR=$USER_BASE/geant4/$GQ_NAME/$GQ_TAG/workdir
	  else 	  
         ##export G4WORKDIR=$HOME/geant4/$GQ_NAME 
         export G4WORKDIR=$USER_BASE/geant4/$GQ_NAME/$GQ_TAG/workdir
	  fi
	  
	  test -d $G4WORKDIR || mkdir -p $G4WORKDIR

      ## revert to using home for use by a non-owner ... thence to user base for batch usage 
      source ${GQ_HOME}/env.sh > $USER_BASE/g4-env.log
      #source ${GQ_HOME}/env.sh > $HOME/g4-env.log
      #source ${GQ_HOME}/env.sh > $GQ_HOME/env.log

      # as are reusing the "dbg" datadir for all tags 
      # NB assumes all the G4data is in one folder
      export GQ_DATA=$(dirname $G4LEDATA)
      export GEANT_CMT="GEANT_incdir:$GQ_HOME/include GEANT_libdir:$GQ_HOME/lib/$G4SYSTEM GEANT_datadir:$GQ_DATA OGLLIBS:newogl:set"
      export ENV2GUI_VARLIST="G4INSTALL:G4SYSTEM:$ENV2GUI_VARLIST"
  
else
	  echo ".bash_geant4_use it seems G4 environment is not set up for use... as cannot find ${GQ_HOME}/env.sh , GQ_HOME:${GQ_HOME:-should-be-defined}"
fi  






if [ "$CMTCONFIG" == "Darwin" ]; then
   export DYLD_LIBRARY_PATH=$G4INSTALL/lib/$G4SYSTEM:$DYLD_LIBRARY_PATH
else 
   export   LD_LIBRARY_PATH=$G4INSTALL/lib/$G4SYSTEM:$LD_LIBRARY_PATH
fi



######## G4 utilities #############################

geant4-env(){
   echo " ======== G4 environment  ==== " 
   env | grep G4 
   echo " ======== GQ environment  ==== " 
   env | grep GQ 
   echo " ======== G4 environment log ($HOME/env.log)  ==== " 
   cat ${HOME}/env.log
}

geant4-copy-examples(){

   cd $G4WORKDIR
   pwd
   cp -rp $GQ_HOME/examples .

##   
##  gives operation not supported (presumably the permission preservation)
##
}

#################  analysis example A01 #######################

_a01(){

   ##  
   ## when 	changing a flag, usually need to do a clean then build
   ##

   echo  G4INSTALL:$G4INSTALL	
   echo  G4WORKDIR:$G4WORKDIR
   cd $G4WORKDIR/examples/extended/analysis/A01
   #make clean G4ANALYSIS_USE=1
   #make       G4ANALYSIS_USE=1
 
# if no aida available... switch off analysis  
   #make clean  G4ANALYSIS_USE= GLOBALLIBS=1
   #make        G4ANALYSIS_USE= GLOBALLIBS=1  
   #make clean  G4ANALYSIS_USE= GLOBALLIBS=1
   make        G4ANALYSIS_USE= 
   
  ## curious seems to still try to use granular
   
}

a01(){

   ############# example running (must be from X11 terminal to work) 
   cd $HOME/geant4/$GQ_NAME/examples/extended/analysis/A01  
   $G4WORKDIR/bin/$G4SYSTEM/A01app 
}

##############################################################


# "-Y" enables X11 forwarding without getting :
# Xlib:  extension "GLX" missing on display "localhost:10.0".
# BUT doesnt fully work ...  no motif or coin/inventor widgets 
alias p-ssh="ssh -Y P" 


