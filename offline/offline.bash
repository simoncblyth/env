
     
 OFFLINE_BASE=$ENV_BASE/offline
 export OFFLINE_HOME=$HOME/$OFFLINE_BASE

 dbi(){ [ -r $OFFLINE_HOME/dbi.bash ]  && . $OFFLINE_HOME/dbi.bash ; }
 

