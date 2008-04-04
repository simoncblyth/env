
#  
#   base environment 
#
#   base-x-pkg
#
#

 

 BASE_NAME=base
 BASE_BASE=$ENV_BASE/$BASE_NAME
 export BASE_BASE

[ "$BASE_DBG" == "1" ] && echo $BASE_BASE.bash

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


base-check-nonzero(){
    local nonzs="$*"
    local err=""
    local ok=""
    for nonz in $nonzs
    do
       local vname=$nonz
       eval vval=\$$nonz
       test -z "$vval" && err="$err base-check-nonzero ERROR : $nonz is of zero length \\n"  || ok="$ok base-check-nonzero OK : $nonz is $vval \\n"
    done
    
    echo $err
}


base-export(){

  local msg="=== $FUNCNAME:"
  local relp=${1:-base}
  local node=${2:-B}
  
  
  local tmp=/tmp/$ENV_BASE/$relp 
  local fold=$(dirname $tmp)  
  ##
  ## cannot export without --force  if the leaf folder exists already , 
  ## but the parent of the leaf must exist
  ##
  
  mkdir -p $fold
  
  local cmd="svn export $HOME/$ENV_BASE/$relp $tmp"
  echo $cmd
  eval $cmd  
  
  
  cmd="scp -r /tmp/$ENV_BASE $node:"
  echo $msg try $cmd      this creates the $ENV_BASE folder at the other end 
}



base-path(){
   perl -e 'require "$ENV{'HOME'}/$ENV{'ENV_BASE'}/base/PATH.pm" ; &PATH::present_var(@ARGV) ; ' $*
}



 [ -r local.bash ]       && . local.bash 
 [ -r batch.bash  ]      && . batch.bash

 
  [ -r ssh-infofile.bash ]  && . ssh-infofile.bash
 
pylibxml2(){  [ -r $HOME/$BASE_BASE/pylibxml2.bash ] && . $HOME/$BASE_BASE/pylibxml2.bash ; } 
libxml2(){    [ -r $HOME/$BASE_BASE/libxml2.bash ]   && . $HOME/$BASE_BASE/libxml2.bash ; }
libxslt(){    [ -r $HOME/$BASE_BASE/libxslt.bash ]   && . $HOME/$BASE_BASE/libxslt.bash ; }
lxml(){       [ -r $HOME/$BASE_BASE/lxml.bash ]      && . $HOME/$BASE_BASE/lxml.bash ; }
test-(){      [ -r $HOME/$BASE_BASE/test.bash ]     && . $HOME/$BASE_BASE/test.bash ; }
patch-(){     [ -r $HOME/$BASE_BASE/patch.bash ]    && . $HOME/$BASE_BASE/patch.bash ; }
system-(){    [ -r $HOME/$BASE_BASE/system.bash ]   && . $HOME/$BASE_BASE/system.bash ; }
network-(){    [ -r $HOME/$BASE_BASE/network.bash ] && . $HOME/$BASE_BASE/network.bash ; }


patch-


 ## caution must exit in same directory as started in 
 cd $base_iwd
 [ -t 0 ] || return 
 [ "$TZERO_DBG" == "1" ]  && echo faked tzero  && return 
 cd $HOME/$BASE_BASE
 
 [ -r alias.bash  ]      && . alias.bash
 [ -r clui.bash  ]       && . clui.bash
 
 [ -r perl.bash ]        && . perl.bash
 [ -r ssh.bash ]         && . ssh.bash
 [ -r ssh-config.bash ]  && . ssh-config.bash
 [ -r tty.bash ]         && . tty.bash
 [ -r cron.bash ]        && . cron.bash
 [ -r service.bash ]     && . service.bash
 [ -r file.bash ]        && . file.bash

 
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




 
