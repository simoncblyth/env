
local-deprecated-usage(){ cat << EOU

EOU
}

local-nodetag-deprecated(){


   ## one letter code that represents the user and node
  
   SUDO=
   CLUSTER_TAG=
   BACKUP_TAG=U
   NODE_NAME=

##  set SUDO to "sudo" if sudo access is needed to create folders / change ownership 
##  in the relevant LOCAL_BASE
##
##     G(blyth@g4pb)   is different as its the source machine
##     G1(blyth@grid1) is different as are trying to run code belonging to dayabaysoft aka "P"
#

if [ "X${_CONDOR_SCRATCH_DIR}" != "X" ]; then
	
   NODE_TAG="G1"
   CLUSTER_TAG="G1" 
   USER=$(whoami)
## as it seems condor env has no USER
   
elif [ "$LOCAL_NODE" == "g4pb" ]; then

   NODE_TAG="G"

elif [ "$LOCAL_NODE" == "coop" ]; then

   NODE_TAG="CO"

elif [ "$LOCAL_NODE" == "cms01" ]; then

   NODE_TAG="C"

elif [ "$LOCAL_NODE" == "gateway" ]; then

   NODE_TAG="B"

elif [ "$LOCAL_NODE" == "g3pb" ]; then

   NODE_TAG="G"
   
elif [ "$LOCAL_NODE" == "dayabay" ]; then

   NODE_TAG="XX"

elif ( [ "$USER" == "dayabaysoft" ] && [ "$LOCAL_NODE" == "grid1" ]); then

   NODE_TAG="P"
   BATCH_TYPE="condor"

elif  [ "$LOCAL_NODE" == "grid1" ]; then

   NODE_TAG="G1"
   NODE_NAME="grid1"
   BATCH_TYPE="condor"
   
elif  [ "${LOCAL_NODE:0:6}" == "albert" ]; then   
   
   NODE_TAG="G1"
   NODE_NAME="grid1"
   BATCH_TYPE="condor"
   
elif  [ "${LOCAL_NODE:0:2}" == "pc" ]; then   
   
   NODE_TAG="N"   
   NODE_NAME="pdsf"
   BATCH_TYPE="SGE"
   
elif (      [ "$USER" == "sblyth" ] && [ "$LOCAL_NODE" == "pal" ]); then

   NODE_TAG="L"

elif (      [ "$USER" == "blyth" ] && [ "$LOCAL_NODE" == "hfag" ]); then

   NODE_TAG="H"
   
elif (      [ "$USER" == "root" ] && [ "$LOCAL_NODE" == "hfag" ]); then

   NODE_TAG="H"

elif (      [ "$USER" == "thho" ] && [ "$LOCAL_NODE" == "thho-laptop" ]); then

   NODE_TAG="T"

elif (      [ "$USER" == "thho" ] && [ "$LOCAL_NODE" == "thho-desktop" ]); then

   NODE_TAG="T"

elif (      [ "$USER" == "thho" ] && [ "$LOCAL_NODE" == "hkvme" ]); then

   NODE_TAG="HKVME"

elif (      [ "$USER" == "exist" ] && [ "$LOCAL_NODE" == "hfag" ]); then

   NODE_TAG="X"
 
elif (     [ "$(uname -n)" == "localhost.localdomain" ]); then

   NODE_TAG="XT" 
                 
else
	
   NODE_TAG="U"

fi


## these are deprecated
export CLUSTER_TAG
export NODE_NAME
export BATCH_TYPE

## these are in heavy usage
export NODE_TAG
export BACKUP_TAG=$(local-backup-tag)
export SUDO=$(local-sudo)



########## TARGET_* specify the remote machine coordinates #####################
#    
#     used as defaults for commands dealing with a remote machine, 
#     define a default remote tag 
#
#
#export TARGET_TAG="P"       ## dayabaysoft@grid1   admin level tasks
#export TARGET_TAG="H"       ##      blyth@hfag     admin level tasks .... yes as is a sudoer
#export TARGET_TAG="X"       ##      exist@hfag     webserver running ....  (not a sudoer)
 export TARGET_TAG="G1"      ##      blyth@grid1    user level tasks ... job submission 


}




local-layout-deprectated(){

export LOCAL_BASE_U=/usr/local
export LOCAL_BASE_G=/usr/local

## export LOCAL_BASE_P=/disk/d4/dayabay/local   ## rozz testing 
export LOCAL_BASE_P=/disk/d3/dayabay/local

export LOCAL_BASE_G1=$DAYABAY_G1/local  
export LOCAL_BASE_L=/usr/local           
export LOCAL_BASE_H=/data/usr/local          
export LOCAL_BASE_T=/usr/local
export LOCAL_BASE_N=$HOME/local
export LOCAL_BASE_C=/data/env/local
export LOCAL_BASE_XT=/home/tianxc

local-base(){
   local tag=${1:-$NODE_TAG} 
   local vname=LOCAL_BASE_$tag
   eval _LOCAL_BASE=\$$vname
   echo ${_LOCAL_BASE:-$LOCAL_BASE_U}
}

export LOCAL_BASE=$(local-base)
export SYSTEM_BASE_U=$LOCAL_BASE
export SYSTEM_BASE_P=$grid1_system_base
export SYSTEM_BASE_G1=$grid1_system_base
export SYSTEM_BASE_C=/data/env/system
export SYSTEM_BASE_XT=/home/tianxc/system
export SYSTEM_BASE_XX=/usr/local

local-system-base(){
   local tag=${1:-$NODE_TAG} 
   local vname=SYSTEM_BASE_$tag
   eval _SYSTEM_BASE=\$$vname
   echo ${_SYSTEM_BASE:-$SYSTEM_BASE_U}
}





if [ "X$DEFAULT_MACRO" == "X" ]; then
  DEFAULT_MACRO="generator-inversebeta_seed-0_angle-0_nevts-100"
#else
#  echo honouring override DEFAULT_MACRO $DEFAULT_MACRO
fi
export DEFAULT_MACRO






}



local-nodeinfo(){

  tags="G P G1 L H U T"
  for t in $tags
  do
     ln=LOCAL_BASE_$t
     un=USER_BASE_$t

	 if [ "$t" == "$NODE_TAG" ]; then
		 mark="==========current=node" 
     elif [ "$t" == "$TARGET_TAG" ]; then
		 mark="==========target=node" 
     else
		 mark="=="
     fi		 
		 
	 eval vln=\$$ln
	 eval vun=\$$un

	 printf "  %-25s  %-50s %-30s %s \n" $t $vln $vun $mark  
  done	  


}

	
local-info(){

  local tags="TAG PORT USER PASS"
  local prefixs="TARGET SCM"
  
  for p in $prefixs
  do	  
  for t in $tags
  do
    n="${p}_${t}"  
	eval v=\$$n
	printf " %-20s %s \n" $n $v 
  done
  done

}










