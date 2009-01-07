

local-usage(){

cat << EOU

  These are used by most functions... and rarely need to be invoked directly by the user

   local-node        :  $(local-node)
   local-nodetag     :  $(local-nodetag)
   local-tag2node    :  $(local-tag2node)
   local-backup-tag  :  $(local-backup-tag)      paired backup node
 
   local-sudo        :  $(local-sudo)            is set on nodes which use system tools mostly
   
   local-system-base :  $(local-system-base)      
   local-base        :  $(local-base)
   local-var-base    :  $(local-var-base)
   local-scm-fold    :  $(local-scm-fold)
   local-user-base   :  $(local-user-base)
   local-output-base :  $(local-output-base)
   
   
   
   local-scm       : define the SCM_* coordinates of source code management node supporting the current node
    
   
     NODE_TAG_OVERRIDE : $NODE_TAG_OVERRIDE
     
     NODE_TAG      : $NODE_TAG
     LOCAL_BASE    : $LOCAL_BASE
     SYSTEM_BASE   : $SYSTEM_BASE     system tools like svn
     VAR_BASE      : $VAR_BASE 
     USER_BASE     : $USER_BASE
     OUTPUT_BASE   : $OUTPUT_BASE
                                
EOU

}

local-env(){

   local dbg=${1:-0}
   local msg="=== $FUNCNAME :"
   
   [ "$dbg" == "1" ] && echo $msg	 
          
   export SOURCE_NODE="g4pb"
   export SOURCE_TAG="G"

   export LOCAL_ARCH=$(uname)
   export LOCAL_NODE=$(local-node)
   export NODE_TAG=$(local-nodetag)       # glean where we are and define NODE_TAG
   export BACKUP_TAG=$(local-backup-tag)  # paired backup for the NODE_TAG 
   export SUDO=$(local-sudo)

   local-scm        # assign coordinates of the SCM server for this node
 
   export SYSTEM_BASE=$(local-system-base) ## prequisite base for most everything, ie where to pick up subversion +
   export LOCAL_BASE=$(local-base)
   export VAR_BASE=$(local-var-base)    ## operational files, like backups
   export SCM_FOLD=$(local-scm-fold)
   export VAR_BASE_BACKUP=$(local-var-base $BACKUP_TAG)
   export USER_BASE=$(local-user-base)
   export OUTPUT_BASE=$(local-output-base)

   [ ! -d "$USER_BASE" ]   && echo "WARNING creating folder USER_BASE $USER_BASE" &&   mkdir -p $USER_BASE 
   [ ! -d "$OUTPUT_BASE" ] && echo "WARNING creating folder OUTPUT_BASE $OUTPUT_BASE" &&   mkdir -p $OUTPUT_BASE 

    local-userprefs

}


local-node(){
   case $NODE in 
     "") echo $(uname -a | cut -d " " -f 2 | cut -d "." -f 1) ;;
      *) echo $NODE ;;
    esac
}

local-userprefs(){
   case $USER in 
     blyth) export SVN_EDITOR=vi ;;
   esac
}


local-tag2node(){
  case ${1:-$NODE_TAG} in 
     H) echo hfag  ;;
     C) echo cms01 ;;
     P) echo grid1 ;;
    G3) echo g3pb ;;
     G) echo g4pb ;; 
     *) echo unknown ;; 
  esac
}

local-nodetag(){
  [ -n "$NODE_TAG_OVERRIDE" ] && echo $NODE_TAG_OVERRIDE && return 0
  case ${1:-$LOCAL_NODE} in
         g4pb) echo G ;;
         coop) echo CO ;;
        cms01) echo C ;;
      gateway) echo B ;;
         g3pb) echo G ;;
          pal) echo L ;;
         hfag) local-nodetag-hfag $USER ;;
  thho-laptop) echo T ;;
 thho-desktop) echo T ;;
        hkvme) echo HKVME ;;
        grid1) local-nodetag-grid1 $USER ;;
            *) local-nodetag-other $(uname -n) ;;
  esac

}

local-nodetag-hfag(){
   case ${1:-$USER} in
      blyth) echo H ;;
      exist) echo X ;;
          *) echo U ;;
   esac
}

local-nodetag-grid1(){
   case ${1:-$USER} in 
     dayabaysoft) echo P ;;
               *) echo G1 ;;
   esac
}

local-nodetag-other(){
   local host=${1:-$(uname -n)}
   if  [ "${host:0:6}" == "albert" ]; then   
        echo G1
   elif [ "${host:0:2}" == "pc" ]; then   
       echo N 
   elif [ "$host" == "dayabay.ihep.ac.cn" ]; then
       local-nodetag-xinchun    
   else
       echo U
   fi
}

local-nodetag-xinchun(){
   case $USER in
     blyth) echo XT ;;
         *) echo XX ;;
   esac
}

local-sudo(){
  case ${1:-$NODE_TAG} in
  G|H|T|C|P) echo sudo ;;
      *) echo -n ;
  esac
}

local-backup-tag(){
   case ${1:-$NODE_TAG} in 
      G) echo G3 ;;
      H) echo C  ;;
      C) echo P  ;;
     XX) echo IHEP C ;;
      *) echo U ;;
   esac  
}

local-email(){
   case ${1:-$NODE_TAG} in
     XX) echo tianxc@ihep.ac.cn ;;
      *) echo blyth@hep1.phys.ntu.edu.tw ;;
   esac
}


local-base(){
    case ${1:-$NODE_TAG} in 
       G) echo /usr/local ;;
      G1) echo /disk/d3/dayabay/local ;;    ## used to be :  /data/w  then /disk/d4
       P) echo /disk/d3/dayabay/local ;;
       L) echo /usr/local ;;
       H) echo /data/usr/local ;;
       T) echo /usr/local ;;
       N) echo $HOME/local ;;
       C) echo /data/env/local ;;
      XT) echo /home/tianxc ;;   
       *) echo /usr/local ;;
   esac
}

local-system-base(){
   case ${1:-$NODE_TAG} in 
      P|G1) echo /disk/d4/dayabay/local ;;
         C) echo /data/env/system ;;
        XT) echo /home/tianxc/system ;;
        XX) echo /usr/local ;;
         *) echo $(local-base $*) ;;
   esac
}

local-var-base(){
   case ${1:-$NODE_TAG} in 
      U) echo /var ;;
      P) echo /disk/d3/var ;;
     G1) echo /disk/d3/var ;;
      N) echo $HOME/var ;;
     XT) echo /home/tianxc ;; 
     XX) echo /home ;; 
   IHEP) echo /home ;;  
      C) echo /var ;;
      *) echo /var ;; 
   esac
}

local-scm-fold(){
   echo $(local-var-base $*)/scm
}

local-user-base(){
   case ${1:-$NODE_TAG} in
      G) echo $HOME/Work ;;
   P|G1) echo /disk/d3/$USER ;;
      L) echo $(local-base L) ;;
      H) echo $(local-base H) ;;
      T) echo $HOME/dybwork ;;
      N) echo $HOME ;;
     XT) echo /home/tianxc ;;  
      *) echo /tmp ;;
   esac
}

local-output-base(){
   case ${1:-$NODE_TAG} in
      N) echo /project/projectdirs/dayabay/scratch/$USER ;;
      *) echo $(local-user-base $*) ;;
   esac
}




	
	
	
	
    
    
local-scm(){


########## SCM_* specify the source code repository coordinates #####################

 #export SCM_TAG="H"       ##      blyth@hfag      trac "production"  
 #export SCM_TAG="G"       ##      blyth@g4pb      trac testing

 # if SCM_TAG is set already use that value, otherwise default to H
 
 private-
 
SCM_TAG=${SCM_TAG:-H}
export SCM_TAG 

if [ "$SCM_TAG" == "P" ]; then
	
   SCM_HOST=grid1.phys.ntu.edu.tw
   SCM_PORT=6060
   SCM_USER=$USER
   SCM_PASS=$(private-val NON_SECURE_PASS)
   SCM_TRAC=env
   SCM_GROUP=GRID1
   
   SCM_URL=http://$SCM_HOST:$SCM_PORT
   
elif [ "$SCM_TAG" == "H" ]; then 

   #SCM_HOST=hfag.phys.ntu.edu.tw
   #SCM_PORT=6060
   SCM_HOST=dayabay.phys.ntu.edu.tw
   SCM_PORT=80
   
   SCM_USER=$USER
   SCM_PASS=$(private-val NON_SECURE_PASS)
   SCM_TRAC=env
   SCM_GROUP=NTU

   SCM_URL=http://$SCM_HOST

elif [ "$SCM_TAG" == "G" ]; then 

   ## trac testing 
   SCM_HOST=localhost
   SCM_PORT=80
   SCM_USER=$USER
   SCM_PASS=$(private-val NON_SECURE_PASS)
   SCM_TRAC=workflow
   SCM_GROUP=DEV
   
   SCM_URL=http://$SCM_HOST
   
else

   SCM_HOST=	
   SCM_PORT=	
   SCM_USER=
   SCM_PASS=
   SCM_GROUP=

fi


export SCM_URL
export SCM_HOST
export SCM_PORT
export SCM_GROUP
export SCM_TRAC





}
    
    
    
    
