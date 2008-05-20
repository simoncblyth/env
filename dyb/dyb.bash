

dyb-usage(){
  
cat << EOU     

  Defining this needed in common to the below ...

   dybi : installation 
   dybr : run time environment setup 
   dybt : testing
   dybx : xecution 
 
EOU


}


dyb-env(){

   DYB_BASE=dyb
   export DYB_HOME=$ENV_HOME/$DYB_BASE

   dyb-params

   export DYB_FOLDER=$LOCAL_BASE/dyb
   export DYB=$DYB_FOLDER/$DYB_VERSION$DYB_OPTION 
   export DYB_RELEASE=NuWa-$DYB_VERSION
   export DDR=$DYB/$DYB_RELEASE 
 
   [ "$NODE_TAG" == "C" ] && DDR=$HOME/NuWa-trunk
   
   
   export DDI=$DYB/installation/$DYB_VERSION/dybinst/scripts   ## should this be fixed at trunk ?

   ## next time distinguish the options (particulary debug on or off status) via the folder name also 

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





dyb(){
   dybr-
   dybr-go $*
}


 






















