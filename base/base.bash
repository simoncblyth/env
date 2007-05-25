
#  
#   base environment 
#
#   base-x-pkg
#
#

 BASE_NAME=base
 BASE_BASE=env/$BASE_NAME
 export BASE_BASE

 iwd=$(pwd)
 cd $HOME/$BASE_BASE

 [ -r local.bash ]       && . local.bash 

 [ -t 0 ] || return 

 [ -r alias.bash  ]      && . alias.bash
 [ -r perl.bash ]        && . perl.bash
 [ -r ssh.bash ]         && . ssh.bash
 [ -r tty.bash ]         && . tty.bash
 [ -r cron.bash ]        && . cron.bash
 [ -r service.bash ]     && . service.bash

 cd $iwd

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


base-datestamp(){

  if [ "$1" == "now" ]; then 
     if [ "$(uname)" == "Darwin" ] ; then
        timdef=$(perl -e 'print time')
	    refdef=$(date -r $timdef +"%Y%m%d")  
     else		
	    refdef=$(date  +"%Y%m%d")
     fi 
  fi  

  echo $refdef 
}

 
