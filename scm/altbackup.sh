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

    SHELL=/bin/bash
    HOME=/home/blyth
    ENV_HOME=/home/blyth/env
    CRONLOG_DIR=/home/blyth/cronlog
    NODE_TAG_OVERRIDE=WW
    MAILTO=blyth@hep1.phys.ntu.edu.tw
    #
    00 13 * * * ( . $ENV_HOME/env.bash ; env- ; python- source ; ssh-- ; $ENV_HOME/scm/altbackup.sh $HOME/cronlog/altbackup.log dump check_source transfer purge_target  ) > $CRONLOG_DIR/altbackup_.log 2>&1

On the receiving **target** node::

    SHELL=/bin/bash
    HOME=/home/blyth
    ENV_HOME=/home/blyth/env
    CRONLOG_DIR=/home/blyth/cronlog
    MAILTO=blyth@hep1.phys.ntu.edu.tw
    #
    30 15 * * * ( . $ENV_HOME/env.bash ; env- ; python- source ; ssh-- ; $ENV_HOME/scm/altbackup.sh $HOME/cronlog/altbackup.log dump check_target ) > $CRONLOG_DIR/altbackup_.log 2>&1



Scheduling
-------------

* ~11:30 scm-backup completes
* 13:00 source node altbackup cron starts 
* ~14:30 source node altbackup completes
* 15:30 target node check starts

The root controlled scm backup (managed by Qiumei) typically completes before noon, as indicated by timestamps on the dna sidecars::

    [dayabay] /home/blyth/e/scm > find /home/scm/backup/dayabay  -name '*.tar.gz.dna' -exec ls -l {} \; | grep dybsvn
    -rw-r--r--  1 root root 65 Feb 26 11:03 /home/scm/backup/dayabay/svn/dybsvn/2013/02/26/104701/dybsvn-19844.tar.gz.dna
    -rw-r--r--  1 root root 65 Feb 25 11:05 /home/scm/backup/dayabay/svn/dybsvn/2013/02/25/104702/dybsvn-19839.tar.gz.dna
    -rw-r--r--  1 root root 65 Feb 23 11:05 /home/scm/backup/dayabay/svn/dybsvn/2013/02/23/104702/dybsvn-19839.tar.gz.dna
    -rw-r--r--  1 root root 65 Feb 24 11:04 /home/scm/backup/dayabay/svn/dybsvn/2013/02/24/104702/dybsvn-19839.tar.gz.dna
    -rw-r--r--  1 root root 64 Feb 26 11:25 /home/scm/backup/dayabay/tracs/dybsvn/2013/02/26/104701/dybsvn.tar.gz.dna
    -rw-r--r--  1 root root 64 Feb 25 11:28 /home/scm/backup/dayabay/tracs/dybsvn/2013/02/25/104702/dybsvn.tar.gz.dna
    -rw-r--r--  1 root root 64 Feb 23 11:28 /home/scm/backup/dayabay/tracs/dybsvn/2013/02/23/104702/dybsvn.tar.gz.dna
    -rw-r--r--  1 root root 64 Feb 24 11:28 /home/scm/backup/dayabay/tracs/dybsvn/2013/02/24/104702/dybsvn.tar.gz.dna
    [dayabay] /home/blyth/e/scm > 



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

