
DYB_BASE=$ENV_BASE/dyb
export DYB_HOME=$HOME/$DYB_BASE

[ -r $DYB_HOME/core.bash ]   && .  $DYB_HOME/core.bash 

[ -r $DYB_HOME/gaudi/gaudi.bash ]   && .  $DYB_HOME/gaudi/gaudi.bash 
[ -r $DYB_HOME/lcgcmt/lcgcmt.bash ] && .  $DYB_HOME/lcgcmt/lcgcmt.bash
[ -r $DYB_HOME/external/external.bash ] && .  $DYB_HOME/external/external.bash




dyb-get(){
   mkdir -p $LOCAL_BASE/dyb
   cd $LOCAL_BASE/dyb
   svn export http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst
}

dyb-install(){
  local def_arg="all"
  local arg=${1:-$def_arg}
  cd $LOCAL_BASE/dyb
  ./dybinst $arg
}


core(){     [ -r $DYB_HOME/core.bash ]          && .  $DYB_HOME/core.bash ; }
lcgcmt(){   [ -r $DYB_HOME/lcgcmt/lcgcmt.bash ] && .  $DYB_HOME/lcgcmt/lcgcmt.bash ; }
dyb(){      [ -r $DYB_HOME/dyb.bash ]           && . $DYB_HOME/dyb.bash ; } 

installation(){     [ -r $DYB_HOME/installation.bash ]          && .  $DYB_HOME/installation.bash ; }

