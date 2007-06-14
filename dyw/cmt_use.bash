[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/cmt_use.bash

##  SUDO removed ... set in base/local.bash

CMT_FOLDER=$LOCAL_BASE/cmt
if ([ "$NODE_TAG" == "G" ] || [ "$NODE_TAG" == "T"  ]); then
	
  CMT_VERS="v1r18p20060606"
  CMT_HOME=${CMT_FOLDER}/CMT${CMT_VERS}/CMT/${CMT_VERS}
  
elif [ "$NODE_TAG" == "L" ]; then

  CMT_VERS="v1r18p20061003"
  CMT_HOME=${CMT_FOLDER}/CMT/${CMT_VERS}

elif ( [ "$NODE_TAG" == "G1" ] || [ "$NODE_TAG" == "P" ] || [ "$NODE_TAG" == "$CLUSTER_TAG" ] ); then

  CMT_VERS="v1r18p20061003"
  CMT_HOME=${CMT_FOLDER}/CMT/${CMT_VERS}


elif [ "$NODE_TAG" == "N" ]; then

  CMT_HOME="external"

else
	
  echo $DYW_BASE/cmt_use.bash NODE_TAG $NODE_TAG CLUSTER_TAG $CLUSTER_TAG not supported 	

fi


if [ "$CMT_HOME" == "external" ]; then
  #echo  cmt is setup externally  
else
  export CMT_HOME 

  if [ -f "$CMT_HOME/mgr/setup.sh" ]; then
    source $CMT_HOME/mgr/setup.sh  > $USER_BASE/cmt-setup.log   
  else
    echo $DYW_BASE/cmt-use.bash it seems that CMT environment not setup, as cannot find $CMT_HOME/mgr/setup.sh CMT_HOME:${CMT_HOME:should-be-defined} 	 
  fi	
fi



cmt-use-log(){
   cat $USER_BASE/mgr/setup.log	
}

cmt-use-env(){
   env | grep CMT
}

cmt-use-info(){

   echo CMT_HOME $CMT_HOME
   echo CMT_VERS $CMT_VERS
   echo SUDO  $SUDO
   echo LOCAL_BASE $LOCAL_BASE
   echo USER_BASE  $USER_BASE
}




