
dyb-env(){
  DYB_BASE=$ENV_BASE/dyb
  export DYB_HOME=$HOME/$DYB_BASE
}

dyb(){  [ -r $DYB_HOME/dyb.bash ]           && . $DYB_HOME/dyb.bash ; }

dyb-version(){

  # note, sensitivity to preset DYB_VERSION ... overrides the below setting
  export DYB_VERSION_P=trunk
  #export DYB_VERSION_P=0.0.4  
  
 if [ "X$DYB_VERSION" == "X" ]; then
   vname=DYB_VERSION_$NODE_TAG
   eval DYB_VERSION=\$$vname
 else
   echo WARNING honouring a preset DYB_VERSION $DYB_VERSION     
 fi
 
 export DYB_OPTION=""
 #export DYB_OPTION="_dbg"
 
 export DYB_VERSION
 export DYB_FOLDER=$LOCAL_BASE/dyb
 export DYB=$DYB_FOLDER/$DYB_VERSION$DYB_OPTION 
 export DYB_RELEASE=NuWa-$DYB_VERSION

 export DDR=$DYB/$DYB_RELEASE 


 ## next time distinguish the options (particulary debug on or off status) via the folder name also 

}

dyb-subs(){
  [ -r $DYB_HOME/dyb-install.bash ] && . $DYB_HOME/dyb-install.bash
  [ -r $DYB_HOME/dyb-run.bash ]     && . $DYB_HOME/dyb-run.bash
}



dyb-env
dyb-version
dyb-subs


















