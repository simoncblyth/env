
[ "$NODE_TAG" == "H" ] && echo not needed here && return 1 
[ "$NODE_TAG" == "U" ] &&  echo not needed here && return 1 


dyb-env(){

## formerly dyb-env

DYB_BASE=dyb
export DYB_HOME=$ENV_HOME/$DYB_BASE

## formerly dyb-version

  # note, sensitivity to preset DYB_VERSION ... overrides the below setting
 
 if [ "X$DYB_VERSION" == "X" ]; then
   vname=DYB_VERSION_$NODE_TAG
   eval DYB_VERSION=\$$vname
 else
   [ "$DYB_DBG" == "1" ] && echo WARNING honouring a preset DYB_VERSION $DYB_VERSION     
 fi
  
 if [ "X$DYB_OPTION" == "X" ]; then
    DYB_OPTION=""
   #DYB_OPTION="_dbg" 
 else
    [ "$DYB_DBG" == "1" ] && echo WARNING honouring a preset DYB_OPTION $DYB_OPTION
 fi
 
 export DYB_OPTION
 export DYB_VERSION
 export DYB_FOLDER=$LOCAL_BASE/dyb
 export DYB=$DYB_FOLDER/$DYB_VERSION$DYB_OPTION 
 export DYB_RELEASE=NuWa-$DYB_VERSION

 export DDR=$DYB/$DYB_RELEASE 
 
 if [ "$NODE_TAG" == "C" ]; then
    DDR=$HOME/NuWa-trunk
 fi
 
 
  ## should this be fixed at trunk ?
 export DDI=$DYB/installation/$DYB_VERSION/dybinst/scripts

 ## next time distinguish the options (particulary debug on or off status) via the folder name also 

 ##
 ## dybi : installation 
 ## dybr : run time environment setup 
 ## dybt : testing
 ## dybx : xecution 
 ##

}



dyb(){   
    
	dybr- 
	dybr-site-setup 
	cd $DDR  
    local loc="$1"
	local qwn="$2"
	
	
	# get rid of positional args to avoid a CMT warning 
    set --
	
	if [ -n "$loc" ]; then
	   echo === dyb : with non blank loc $loc
	   if [ -d "$loc" ]; then
	      echo === dyb : tis a folder
	      cd $loc
		elif [ -f "$loc"  ]; then
		  echo === dyb : tis a file 
		  cd $(dirname $loc)
		fi
	fi
	
	if [ "$(basename $PWD)" == "cmt" ]; then
	  if [ ! -f "setup.sh" ]; then
		 echo === dyb : doing cmt config as in cmt folder with no setup.sh
		 cmt config
	  fi
	  if [ -f "setup.sh" ]; then
	     echo === dyb : sourcing setup
		 . setup.sh
	  else
	     echo === dyb : ERROR no setup.sh in cmt folder after cmt config
	  fi
	fi
	
	if [ "$qwn" != "" ]; then
	    echo === dyb : non blank qwn $qwn : [ cmt show value $qwn ] in $PWD
	    cmt show macro_value $qwn 
	else
	    echo === dyb : blank qwn in $PWD 
	fi
	
	
	# pwd
 }
 


dybr-(){  [ -r $DYB_HOME/dybr.bash ] && . $DYB_HOME/dybr.bash ; }
dybi-(){  [ -r $DYB_HOME/dybi.bash ] && . $DYB_HOME/dybi.bash ; }
dybt-(){  [ -r $DYB_HOME/dybt.bash ] && . $DYB_HOME/dybt.bash ; }





















