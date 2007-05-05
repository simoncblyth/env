
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
   
elif (         [ "$USER" == "blyth" ] && [ "$LOCAL_NODE" == "g4pb" ]); then

   NODE_TAG="G"
   
elif ( [ "$USER" == "dayabaysoft" ] && [ "$LOCAL_NODE" == "grid1" ]); then

   NODE_TAG="P"

elif (       [ "$USER" == "blyth" ] && [ "$LOCAL_NODE" == "grid1" ]); then

   NODE_TAG="G1"
   
elif (      [ "$USER" == "sblyth" ] && [ "$LOCAL_NODE" == "pal" ]); then

   NODE_TAG="L"

elif (      [ "$USER" == "blyth" ] && [ "$LOCAL_NODE" == "hfag" ]); then

   NODE_TAG="H"
   SUDO="sudo"

elif (      [ "$USER" == "thho" ] && [ "$LOCAL_NODE" == "thho-laptop" ]); then

   NODE_TAG="T"
   SUDO="sudo"

elif (      [ "$USER" == "exist" ] && [ "$LOCAL_NODE" == "hfag" ]); then

   NODE_TAG="X"
     
else
	
   NODE_TAG="U"

fi

export NODE_TAG
export CLUSTER_TAG
export SUDO




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

 export SCM_TAG="H"       ##      blyth@hfag     admin level tasks .... yes as is a sudoer

if [ "$SCM_TAG" == "P" ]; then
	
   SCM_HOST=grid1.phys.ntu.edu.tw
   SCM_PORT=6060
   SCM_USER=$USER
   SCM_PASS=$NON_SECURE_PASS
   
elif [ "$SCM_TAG" == "H" ]; then 

   SCM_HOST=hfag.phys.ntu.edu.tw
   SCM_PORT=6060
   SCM_USER=$USER
   SCM_PASS=$NON_SECURE_PASS

else

   SCM_HOST=	
   SCM_PORT=	
   SCM_USER=
   SCM_PASS=

fi

export SCM_HOST
export SCM_PORT

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






## --------------  the root for most everything ---------------------------
##  

export LOCAL_BASE_G=/usr/local
export LOCAL_BASE_P=/disk/d4/dayabay/local   ## must match the setting in P:.bash_profile 
export LOCAL_BASE_G1=/disk/d4/dayabay/local  ## must match the setting in P:.bash_profile 
export LOCAL_BASE_L=/usr/local               ## must match the setting in L:.bash_profile 
export LOCAL_BASE_U=/usr/local               ## must match the setting in L:.bash_profile 
export LOCAL_BASE_H=/data/usr/local          ## must match the setting in L:.bash_profile 
export LOCAL_BASE_T=/usr/local/simon

vname=LOCAL_BASE_$NODE_TAG
eval LOCAL_BASE=\$$vname
export LOCAL_BASE

## -------------- user specific base , for users macros and job outputs ----------

export USER_BASE_G=$HOME/Work
export USER_BASE_P=/disk/d4/$USER
export USER_BASE_G1=/disk/d4/$USER  
export USER_BASE_L=$LOCAL_BASE_L
export USER_BASE_U=$LOCAL_BASE_U
export USER_BASE_H=$LOCAL_BASE_H
export USER_BASE_T=$HOME/simon

vname=USER_BASE_$NODE_TAG
eval USER_BASE=\$$vname
export USER_BASE




	
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

	
