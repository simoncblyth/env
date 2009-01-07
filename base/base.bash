

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
 

base-env(){

  local dbg=${1:-0}
  local iwd=$(pwd)
 

   cd $(env-home)/base
   elocal-
 
   
   ## do not need the ssh- funcs when non-interactive but do need the connection to the agent 
   ##  so this is better separate from the ssh-
   [ -r ssh-infofile.bash ]  && . ssh-infofile.bash

   ## caution must exit in same directory as started in 
   cd $iwd
   
   [ -t 0 ] || return 
   [ "$dbg" == "t0fake" ]  && echo faked tzero  && return 
 
   cd $(env-home)/base
   
   clui-
   ssh-
 
   cd $iwd
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





 
