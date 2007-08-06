
[ "$BASE_DBG" == "1" ] && echo local.bash

 ## if NODE is set use that otherwise determine from uname  

export LOCAL_ARCH=$(uname)

if [ "X$NODE" == "X" ]; then
    LOCAL_NODE=$(uname -a | cut -d " " -f 2 | cut -d "." -f 1)	 
else
    LOCAL_NODE=$NODE
fi 	 
export LOCAL_NODE 

export SOURCE_NODE="g4pb"
export SOURCE_TAG="G"


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
   BACKUP_TAG="G3"
   SUDO="sudo"
   
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
   BACKUP_TAG="P"
   SUDO="sudo"
   
elif (      [ "$USER" == "root" ] && [ "$LOCAL_NODE" == "hfag" ]); then

   NODE_TAG="H"
   BACKUP_TAG="P"
   SUDO=""

elif (      [ "$USER" == "thho" ] && [ "$LOCAL_NODE" == "thho-laptop" ]); then

   NODE_TAG="T"
   SUDO="sudo"


elif (      [ "$USER" == "thho" ] && [ "$LOCAL_NODE" == "thho-desktop" ]); then

   NODE_TAG="T"
   SUDO="sudo"



elif (      [ "$USER" == "exist" ] && [ "$LOCAL_NODE" == "hfag" ]); then

   NODE_TAG="X"
     
else
	
   NODE_TAG="U"

fi

export NODE_TAG
export BACKUP_TAG
export CLUSTER_TAG
export SUDO
export NODE_NAME
export BATCH_TYPE



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


########## SCM_* specify the source code repository coordinates #####################

 #export SCM_TAG="H"       ##      blyth@hfag      trac "production"  
 #export SCM_TAG="G"       ##      blyth@g4pb      trac testing

 # if SCM_TAG is set already use that value, otherwise default to H
 
SCM_TAG=${SCM_TAG:-H}
export SCM_TAG 

if [ "$SCM_TAG" == "P" ]; then
	
   SCM_HOST=grid1.phys.ntu.edu.tw
   SCM_PORT=6060
   SCM_USER=$USER
   SCM_PASS=$NON_SECURE_PASS
   SCM_TRAC=env
   SCM_GROUP=GRID1
   
elif [ "$SCM_TAG" == "H" ]; then 

   SCM_HOST=hfag.phys.ntu.edu.tw
   SCM_PORT=6060
   SCM_USER=$USER
   SCM_PASS=$NON_SECURE_PASS
   SCM_TRAC=env
   SCM_GROUP=NTU

elif [ "$SCM_TAG" == "G" ]; then 

   ## trac testing 
   SCM_HOST=localhost
   SCM_PORT=80
   SCM_USER=$USER
   SCM_PASS=$NON_SECURE_PASS
   SCM_TRAC=workflow
   SCM_GROUP=DEV

else

   SCM_HOST=	
   SCM_PORT=	
   SCM_USER=
   SCM_PASS=
   SCM_GROUP=

fi

export SCM_HOST
export SCM_PORT
export SCM_GROUP
export SCM_TRAC

##################################################################################





base-node-info(){

  tags="TAG PORT USER PASS"
  prefixs="TARGET SCM"
  
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


#export DISK_G1=/data/w
export DISK_G1=/disk/d4
export DAYABAY_G1=$DISK_G1/dayabay


## --------------  the software root for most everything ---------------------------
##  

export LOCAL_BASE_U=/tmp
export LOCAL_BASE_G=/usr/local
export LOCAL_BASE_P=$DAYABAY_G1/local   
export LOCAL_BASE_G1=$DAYABAY_G1/local  
export LOCAL_BASE_L=/usr/local           
export LOCAL_BASE_H=/data/usr/local          
export LOCAL_BASE_T=/usr/local/dyw
export LOCAL_BASE_N=$HOME/local

vname=LOCAL_BASE_$NODE_TAG
eval _LOCAL_BASE=\$$vname
export LOCAL_BASE=${_LOCAL_BASE:-$LOCAL_BASE_U}

## ----------  for operational files, like backups

export VAR_BASE_U=/var
export VAR_BASE_P=$DAYABAY_G1/var
export VAR_BASE_G1=$DAYABAY_G1/var
export VAR_BASE_G3=/var
export VAR_BASE_H=/var
export VAR_BASE_G=/var
export VAR_BASE_N=$HOME/var

## if a value for the node is defined then use that, otherwise use VAR_BASE_U
vname=VAR_BASE_$NODE_TAG
eval _VAR_BASE=\$$vname
export VAR_BASE=${_VAR_BASE:-$VAR_BASE_U}


## ------------- path on remote backup machine 
vname=VAR_BASE_$BACKUP_TAG 
eval _VAR_BASE_BACKUP=\$$vname
export VAR_BASE_BACKUP=${_VAR_BASE_BACKUP:-$VAR_BASE_U}

## -------------- user specific base , for users macros and job outputs ----------


export USER_BASE_U=/tmp
export USER_BASE_G=$HOME/Work
export USER_BASE_P=$DISK_G1/$USER
export USER_BASE_G1=$DISK_G1/$USER  
export USER_BASE_L=$LOCAL_BASE_L
export USER_BASE_H=$LOCAL_BASE_H
export USER_BASE_T=$HOME/dybwork
export USER_BASE_N=$HOME

## if a value for the node is defined then use that, otherwise use VAR_BASE_U
vname=USER_BASE_$NODE_TAG
eval _USER_BASE=\$$vname
export USER_BASE=${_USER_BASE:-$USER_BASE_U}





if [ "X$DEFAULT_MACRO" == "X" ]; then
  DEFAULT_MACRO="generator-inversebeta_seed-0_angle-0_nevts-100"
else
  echo honouring override DEFAULT_MACRO $DEFAULT_MACRO
fi
export DEFAULT_MACRO



## --------------  for job outputs 

# export OUTPUT_BASE_G=/tmp
# export OUTPUT_BASE_P=/tmp
# export OUTPUT_BASE_G1=/tmp
# vname=OUTPUT_BASE_$NODE_TAG
# eval OUTPUT_BASE=\$$vname

export OUTPUT_BASE_U=$USER_BASE
export OUTPUT_BASE_N=/project/projectdirs/dayabay/scratch/blyth
vname=OUTPUT_BASE_$NODE_TAG
eval _OUTPUT_BASE=\$$vname
export OUTPUT_BASE=${_OUTPUT_BASE:-$OUTPUT_BASE_U}




[ -d "$USER_BASE" ] || ( echo "WARNING creating folder USER_BASE $USER_BASE" &&   mkdir -p $USER_BASE )
[ -d "$OUTPUT_BASE" ] || ( echo "WARNING creating folder OUTPUT_BASE $OUTPUT_BASE" &&   mkdir -p $OUTPUT_BASE )

	
base-info(){

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

	
