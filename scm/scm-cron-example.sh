#! /bin/bash 
# backup svn & trac daily 

export HOME=/home/blyth
export ENV_HOME=$HOME/env 

env-(){ . $HOME/env/env.bash && env-env ; }  

env-
scm-backup- 
type scm-backup-all 

scm-backup-all


exit 0 


