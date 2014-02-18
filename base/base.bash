base-vi(){ vi $(env-home)/base/base.bash ; }

pylibxml2-(){ [ -r $ENV_HOME/base/pylibxml2.bash ] && . $ENV_HOME/base/pylibxml2.bash ; } 
libxml2-(){   [ -r $ENV_HOME/base/libxml2.bash ]   && . $ENV_HOME/base/libxml2.bash ; }
libxslt-(){   [ -r $ENV_HOME/base/libxslt.bash ]   && . $ENV_HOME/base/libxslt.bash ; }
lxml-(){      [ -r $ENV_HOME/base/lxml.bash ]      && . $ENV_HOME/base/lxml.bash ; }

patch-(){     [ -r $ENV_HOME/base/patch.bash ]     && . $ENV_HOME/base/patch.bash ; }
system-(){    [ -r $ENV_HOME/base/system.bash ]    && . $ENV_HOME/base/system.bash ; }
network-(){   [ -r $ENV_HOME/base/network.bash ]   && . $ENV_HOME/base/network.bash ; }

clui-(){      . $(env-home)/base/clui.bash    && clui-env $* ; }
cron-(){      . $(env-home)/base/cron.bash    && cron-env $* ; }
service-(){   . $ENV_HOME/base/service.bash && service-env $* ; }
cluster-(){   . $ENV_HOME/base/cluster.bash && cluster-env $* ; }
perl-(){      . $ENV_HOME/base/perl.bash    && perl-env $* ; }
batch-(){     . $ENV_HOME/base/batch.bash   && batch-env $* ; }
file-(){      . $ENV_HOME/base/file.bash    && file-env $* ; }

ssh-(){        . $(env-home)/base/ssh.bash    && ssh--env $* ; }
ssh--(){       . $(env-home)/base/ssh.bash    && ssh--env $* ; }
 

base-usage(){
   cat << EOU

       base-datestamp
       base-pathstamp 

EOU


}


base-env(){

  local dbg=${1:-0}
  local iwd=$(pwd)
  local sshinfo=$(env-home)/base/ssh-infofile.bash

   elocal-
 
   ## do not need the ssh- funcs when non-interactive but do need the connection to the agent 
   ##  so this is better separate from the ssh-

   ssh--

   source $sshinfo 

   #case $NODE_TAG in 
   #        D) ssh--osx-keychain-sock-export ;;
   #        *) echo sshinfo $sshinfo && source $sshinfo ;;   ## HUH why the split
   #esac

   [ -t 0 ] || return 
   [ "$dbg" == "t0fake" ]  && echo faked tzero  && return 
 
   clui-
 
   #cd $iwd
}

 
 
 
 
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

base-stat(){
   local path=$1
   case $(uname) in
     Darwin) stat -f "%m" $path ;; 
      Linux) stat -c "%Z" $path ;;
   esac
}


base-pathstamp(){
   local path=$1
   local fmt=${2:-"%Y%m%d"}
   [ ! -f $path ] && return 0
   case $(uname) in 
      Darwin) date -r $(base-stat $path) +$fmt  ;;
       Linux) date -r $path +$fmt ;;
   esac
}



base-rln(){
    local base=$1
    local name=$2
    local lnk=$base/$name ;
    if [ -L "$lnk" ]; then 
       local tgt=$(readlink $lnk)
       echo $tgt
    fi
}

base-ln(){
    local msg="=== $FUNCNAME :";

    local base=$1
    local arg=$2
    [ -z "$arg" ] && base-ln-ls $base && return $?


    local path
    func-
    func-isfunc- $arg && path=$($arg) || path=$arg

    #echo arg $arg path $path
    #return 0
    local name=${3:-$(basename $path)}

    [ ! -d "$path" ] && echo $msg ABORT no such path $path && return 1
    local lnk=$base/$name ;
    local cmd

    if [ -L "$lnk" ]; then 
       local tgt=$(readlink $lnk)
       if [ "$tgt" == "$path" ]; then
           echo $msg link $lnk already points to $path && return 0
       else
           echo $msg old link $lnk points to $tgt ... changing to $path
           cmd="sudo rm $base/$name ; sudo ln -sf $path $base/$name";
       fi
    else
       echo $msg creating new link $lnk to $path 
       cmd="sudo ln -s $path $base/$name";
    fi

    echo $msg $cmd ... lnk $lnk ;
    eval $cmd
}


base-ln-ls(){
  local base=$1
  local item 
  for item in $base/* ; do
    if [ -L "$item" ]; then
       local lnk=$item
       local name=$(basename $lnk)
       local tgt=$(readlink $lnk)
       local sta
       [ ! -d "$tgt" ] && sta="MISSING" || sta=""
       
       printf " %-15s %-10s %s \n" $name $sta $tgt 
    fi
  done
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




base-path(){
   perl -e 'require "$ENV{'ENV_HOME'}/base/PATH.pm" ; &PATH::present_var(@ARGV) ; ' $*
}





 
