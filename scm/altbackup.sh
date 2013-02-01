#!/bin/bash

altbackup_usage(){ cat << EOU

Simple scp based tarball transfer and checking script
=======================================================

NB this does not actually do the backup, the ancient scm-backup machinery
is still doing that 

::

     $ENV_HOME/scm/altbackup.sh $HOME/cronlog/altbackup.log dump check_source transfer purge_target


Crontab examples 
-----------------

On the sending **source** node::

	16 17 * * * ( . $ENV_HOME/env.bash ; env- ; python- source ; ssh-- ; $ENV_HOME/scm/altbackup.sh $HOME/cronlog/altbackup.log dump check_source transfer purge_target  ) > $CRONLOG_DIR/altbackup_.log 2>&1

On the receiving **target** node::

	16 18 * * * ( . $ENV_HOME/env.bash ; env- ; python- source ; ssh-- ; $ENV_HOME/scm/altbackup.sh $HOME/cronlog/altbackup.log dump check_target ) > $CRONLOG_DIR/altbackup_.log 2>&1

Notification
-------------

In order to be notified incase of non-zero return codes from the scripts
the MAILTO envvar needs to be defined to email addresses in the crontab.


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

