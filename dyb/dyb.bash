
DYB_BASE=$ENV_BASE/dyb
export DYB_HOME=$HOME/$DYB_BASE
export DYB=$LOCAL_BASE/dyb

[ -r $DYB_HOME/core.bash ]   && .  $DYB_HOME/core.bash 

[ -r $DYB_HOME/gaudi/gaudi.bash ]   && .  $DYB_HOME/gaudi/gaudi.bash 
[ -r $DYB_HOME/lcgcmt/lcgcmt.bash ] && .  $DYB_HOME/lcgcmt/lcgcmt.bash
[ -r $DYB_HOME/external/external.bash ] && .  $DYB_HOME/external/external.bash


dyb-get(){
   mkdir -p $LOCAL_BASE/dyb
   cd $LOCAL_BASE/dyb
   svn export http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst
}


dyb-install-nohup(){
    cd $LOCAL_BASE/dyb
    rm -f nohup.out
    nohup bash -lc dyb-install $*
}

dyb-install(){

  local def_arg="all"
  local arg=${1:-$def_arg}
  cd $LOCAL_BASE/dyb
  ./dybinst $arg
}


dyb-sleep(){
  sleep $* && echo "dyb-sleep completed $* " > /tmp/dyb-sleep
}  
  
dyb-smry(){
  cd $LOCAL_BASE/dyb
  tail -f nohup.out
}  
  
dyb-log(){
  cd $LOCAL_BASE/dyb
  tail -f dybinst.log
}  




core(){     [ -r $DYB_HOME/core.bash ]          && .  $DYB_HOME/core.bash ; }
lcgcmt(){   [ -r $DYB_HOME/lcgcmt/lcgcmt.bash ] && .  $DYB_HOME/lcgcmt/lcgcmt.bash ; }
dyb(){      [ -r $DYB_HOME/dyb.bash ]           && . $DYB_HOME/dyb.bash ; } 

installation(){     [ -r $DYB_HOME/installation.bash ]          && .  $DYB_HOME/installation.bash ; }

