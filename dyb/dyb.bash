

dyb-usage(){
  
cat << EOU     

  Defining this needed in common to the below ...

   dybi : installation 
   dybr : run time environment setup 
   dybt : testing
   dybx : xecution 
 
   dyb-bv-setup   : define Bretts dyb function allowing rapid jumping into CMT env
                     eg     
                            dyb GenTools 
                            dyb TrigSim
                            dyb ElecSim 
 
EOU


}


dyb-bv-setup(){

   local rc=$HOME/.dybrc
   
   local dybsh=$DDR/dybgaudi/Utilities/Shell/bash/dyb.sh 
   
   [ ! -f $dybsh ] && return 

   
   if [ ! -f $rc ]; then
      cat << EOR > $rc
## caution this DYB_RELEASE is the full path not the same as my DYB__RELEASE which moved from DYB_RELEASE to avoid name clash
export DYB_RELEASE=$DDR
do_setup=yes
EOR
       echo $msg created $rc 
   else
       echo $msg using preexisting $rc   > /dev/null
   fi
   #cat $rc

   . $dybsh 

}



dyb-env(){

   DYB_BASE=dyb
   export DYB_HOME=$ENV_HOME/$DYB_BASE

   dyb-params
   if [ -z "$NUWA_HOME" ]; then
      export DYB_FOLDER=$LOCAL_BASE/dyb
      export DYB=$DYB_FOLDER/$DYB_VERSION$DYB_OPTION 
   else
      ## base active version on a single envvar NUWA_HOME ... set in .bash_profile 
      export DYB=$(dirname $NUWA_HOME)
      export DYB_VERSION=${NUWA_VERSION:-trunk}
      export DYB_FOLDER=$(dirname $DYB)
   fi
 
   export DYB__RELEASE=NuWa-$DYB_VERSION
   export DDR=$DYB/$DYB__RELEASE 
 
 
 
   [ "$NODE_TAG" == "COOP" ] && DDR=$HOME/NuWa-trunk
   
    
   export DDI=$DYB/installation/$DYB_VERSION/dybinst/scripts   ## should this be fixed at trunk ?
   export DDU=$DDR/dybgaudi/Utilities/Shell/bash
   export DDT=$DYB/installation/$DYB_VERSION/dybtest 
   
   case $NODE_TAG in 
     P|G1|G|C) export NUWA_HOME=$DDR ;;
            *) echo -n ;;
   esac  


   ## next time distinguish the options (particulary debug on or off status) via the folder name also 

   
   dyb_hookup $DYB


   dyb-bv-setup

}


dyb_scripts(){ echo installation/trunk/dybtest/scripts ; }


dyb_hookup(){

    local msg="=== $FUNCNAME : "
    local base=$1
    local dyb__=$base/$(dyb_scripts)/dyb__.sh
    local slave=$base/$(dyb_scripts)/slave.bash
    
    #echo $msg base $base dyb__ $dyb__ slave $slave 
    
    if [ -f $dyb__ ]; then
        . $dyb__
        [ -z $BASH_SOURCE ] && eval "function dyb__source(){  echo $dyb__ ; }"      ## workaround for older bash  
        dyb__default(){ echo dybgaudi/Simulation/GenTools ; } 
    else
        echo $msg no dyb__ $dyb__
    fi 
     
    if [ -f $slave ]; then
       . $slave
    else
       echo $msg no slave $slave 
    fi
     
}




dyb-find(){
   local msg="=== $FUNCNAME :"
   local cmd="mdfind -onlyin $DDR $* | grep -v .svn"
   echo $msg $cmd  ... suspect this fails to look inside requirements files 
   eval $cmd
}

dyb-rfind(){

   local msg="=== $FUNCNAME :"
   local cmd="find $DDR -name requirements -exec grep -H $* {} \; "
   echo $cmd
   eval $cmd 

}



dyb-params(){

    # 
    # note determine DYB_VERSION/OPTION in order, 1st succeeds 
	#      1) preset DYB_VERSION/OPTION  
	#      2) preset DYB_VERSION/OPTION_<NODE_TAG>
    #      3) defaults defined here
	#

   local vname
 
   if [ "X$DYB_VERSION" == "X" ]; then
      vname=DYB_VERSION_$NODE_TAG
      eval DYB_VERSION=\$$vname
	  if [ -z $DYB_VERSION ]; then
		  DYB_VERSION=trunk   
		  [ "$DYB_DBG" == "1" ] && echo WARNING no DYB_VERSION setting for NODE_TAG $NODE_TAG ... assumed default DYB_VERSION $DYB_VERSION 
      fi 
   else
      [ "$DYB_DBG" == "1" ] && echo WARNING honouring a preset DYB_VERSION $DYB_VERSION     
   fi
  
   if [ "X$DYB_OPTION" == "X" ]; then
       
	  vname=DYB_OPTION_$NODE_TAG
      eval DYB_OPTION=\$$vname
	  if [ -z $DYB_OPTION ]; then
		  DYB_OPTION="_dbg"   
		  [ "$DYB_DBG" == "1" ] && echo WARNING no DYB_OPTION setting for NODE_TAG $NODE_TAG ... assumed default DYB_OPTION $DYB_OPTION 
      fi 

   else
       [ "$DYB_DBG" == "1" ] && echo WARNING honouring a preset DYB_OPTION $DYB_OPTION
   fi
 
   export DYB_OPTION
   export DYB_VERSION

}






dyb--(){

   echo $msg THIS IS DEPRECATED USE BRETTS ONE ... ITS MUCH BETTER  
   dybr-
   dybr-go $*
}


 






















