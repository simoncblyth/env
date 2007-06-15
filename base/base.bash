
#  
#   base environment 
#
#   base-x-pkg
#
#

 BASE_NAME=base
 BASE_BASE=$ENV_BASE/$BASE_NAME
 export BASE_BASE

 base_iwd=$(pwd)
 cd $HOME/$BASE_BASE


## needed from cron cmdline so must be before the "-t 0" cutoff 
base-datestamp(){
  local moment=${1:-"now"} 
  local fmt=${2:-"%Y%m%d"}
  if [ "$moment" == "now" ]; then 
     if [ "$(uname)" == "Darwin" ] ; then
        timdef=$(perl -e 'print time')
	    refdef=$(date -r $timdef +$fmt )  
     else		
	    refdef=$(date  +$fmt)
     fi 
  fi  
  echo $refdef 
}


 [ -r local.bash ]       && . local.bash 

 SSH_INFOFILE=$HOME/.ssh-agent-info-$NODE_TAG
 export SSH_INFOFILE
 [ -r $SSH_INFOFILE ]   && . $SSH_INFOFILE

 ## caution must exit in same directory as started in 
 cd $base_iwd
 [ -t 0 ] || return 


 cd $HOME/$BASE_BASE
 
 [ -r alias.bash  ]      && . alias.bash
 [ -r perl.bash ]        && . perl.bash
 [ -r ssh.bash ]         && . ssh.bash
 [ -r tty.bash ]         && . tty.bash
 [ -r cron.bash ]        && . cron.bash
 [ -r service.bash ]     && . service.bash
 [ -r file.bash ]        && . file.bash
 [ -r batch.bash  ]      && . batch.bash
 
  if ([ "$NODE_TAG" == "G1" ] || [ "$NODE_TAG" == "P" ]) then
     [ -r cluster.bash ] && . cluster.bash
  fi
 

 cd $base_iwd

 base-x-pkg(){ 
   cd $HOME 	
   tar zcvf $BASE_NAME.tar.gz $BASE_BASE/*
   scp $BASE_NAME.tar.gz ${1:-$TARGET_TAG}:; 
   ssh ${1:-$TARGET_TAG} "tar zxvf $BASE_NAME.tar.gz" 
 }

 base-i(){
   [ -r $HOME/$BASE_BASE/$BASE_NAME.bash ] && .  $HOME/$BASE_BASE/$BASE_NAME.bash 
 }

 base-vi(){
    iwd=$(pwd)
	cd $HOME/$BASE_BASE
	vi *.bash
	cd $iwd
 }




 
