
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
  cd $LOCAL_BASE/dyb
  ./dybinst all 
}



core(){     [ -r $DYB_HOME/core.bash ]          && .  $DYB_HOME/core.bash ; }
lcgcmt(){   [ -r $DYB_HOME/lcgcmt/lcgcmt.bash ] && .  $DYB_HOME/lcgcmt/lcgcmt.bash ; }




