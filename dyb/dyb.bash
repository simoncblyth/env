

## formerly dyb-env

DYB_BASE=$ENV_BASE/dyb
export DYB_HOME=$HOME/$DYB_BASE

## formerly dyb-version

  # note, sensitivity to preset DYB_VERSION ... overrides the below setting
 
 if [ "X$DYB_VERSION" == "X" ]; then
   vname=DYB_VERSION_$NODE_TAG
   eval DYB_VERSION=\$$vname
 else
   echo WARNING honouring a preset DYB_VERSION $DYB_VERSION     
 fi
  
 if [ "X$DYB_OPTION" == "X" ]; then
    DYB_OPTION=""
   #DYB_OPTION="_dbg" 
 else
    echo WARNING honouring a preset DYB_OPTION $DYB_OPTION
 fi
 
 export DYB_OPTION
 export DYB_VERSION
 export DYB_FOLDER=$LOCAL_BASE/dyb
 export DYB=$DYB_FOLDER/$DYB_VERSION$DYB_OPTION 
 export DYB_RELEASE=NuWa-$DYB_VERSION

 export DDR=$DYB/$DYB_RELEASE 
 
  ## should this be fixed at trunk ?
 export DDI=$DYB/installation/$DYB_VERSION/dybinst/scripts

 ## next time distinguish the options (particulary debug on or off status) via the folder name also 

dyb-(){   [ -r $DYB_HOME/dyb.bash ]  && . $DYB_HOME/dyb.bash ; }
dybr-(){  [ -r $DYB_HOME/dybr.bash ] && . $DYB_HOME/dybr.bash ; }
dybi-(){  [ -r $DYB_HOME/dybi.bash ] && . $DYB_HOME/dybi.bash ; }
dybt-(){  [ -r $DYB_HOME/dybt.bash ] && . $DYB_HOME/dybt.bash ; }





















