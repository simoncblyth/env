#!/bin/bash

altbackup_usage(){ cat << EOU
Usage
=======

::

     $ENV_HOME/scm/altbackup.sh $HOME/cronlog/altbackup.log dump check_source transfer purge_target


EOU
}

altbackup_notify(){
   local msg="=== $FUNCNAME:"
   local logpath=$1
   local subject="$msg FAILURE $(date) $logpath $(hostname) "
   [ -z "$MAILTO" ] && echo $subject : NEED TO SET MAILTO envvar for notification emails && return
   echo $subject : sending notification MAILTO $MAILTO 
   mail -s "$subject" "$MAILTO" < $logpath
}
altbackup_main(){
   local msg="=== $FUNCNAME:"
   local logpath=$1
   shift 
   local cmd="$ENV_HOME/scm/altbackup.py -o $logpath $*"

   echo $msg $cmd
   echo $msg log truncate $(date) > $logpath
   $cmd
   RC=$?

   if [ $RC -ne 0 ]; then  
       echo $msg ERROR RC $RC
       altbackup_notify $logpath 
   else
       echo $msg completed without error RC $RC    
   fi

}


[ $# -eq 0 ] && altbackup_usage && exit 0
altbackup_main $*

