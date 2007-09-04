
DYB_BASE=$ENV_BASE/dyb
export DYB_HOME=$HOME/$DYB_BASE

[ -r $DYB_HOME/core.bash ]   && .  $DYB_HOME/core.bash 

[ -r $DYB_HOME/gaudi/gaudi.bash ]   && .  $DYB_HOME/gaudi/gaudi.bash 
[ -r $DYB_HOME/lcgcmt/lcgcmt.bash ] && .  $DYB_HOME/lcgcmt/lcgcmt.bash
[ -r $DYB_HOME/external/external.bash ] && .  $DYB_HOME/external/external.bash






