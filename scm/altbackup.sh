#!/bin/bash

altbackup_usage(){ cat << EOU

bash wrapper for *altbackup.py*
==================================

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


Current Observed Timings, May 2013
------------------------------------

#. 10:47 backups started on source node
#. 12:00 root (Qiumei) controlled scm backup typically completes before noon, as indicated by timestamps on the dna sidecars on source node
#. 13:00 source node altbackup starts
#. 13:00 svn transfers started  
#. 13:40~13:50 svn transfers completed
#. 13:40~13:50 trac transfers started
#. 14:00~14:20 trac transfers completed, indicated by timestamps on dna sidecars on target node 
#. 14:30 source node altbackup completes
#. 15:40 target node check starts

::

	[blyth@cms01 ~]$ altbackup.py ls
	2013-05-20 11:28:00,183 env.scm.altbackup INFO     /data/env/local/env/home/bin/altbackup.py ls
	2013-05-20 11:28:00,184 env.scm.altbackup INFO     interpreted day string None into 2013/05/20 
	2013-05-20 11:28:00,185 env.scm.altbackup INFO     ================================ ls 
	2013-05-20 11:28:00,185 env.scm.altbackup INFO     find /data/var/scm/alt.backup/dayabay -name '*.tar.gz' -exec ls -lh {} \;
	2013-05-20 11:28:00,231 env.scm.altbackup INFO     
	-rw-r--r--  1 blyth blyth 2.4G May 17 13:51 /data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/05/17/104702/dybsvn-20550.tar.gz
	-rw-r--r--  1 blyth blyth 2.4G May 18 13:37 /data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/05/18/104702/dybsvn-20557.tar.gz
	-rw-r--r--  1 blyth blyth 2.4G May 19 13:38 /data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/05/19/104702/dybsvn-20561.tar.gz
	-rw-r--r--  1 blyth blyth 1.5G May 17 14:20 /data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/05/17/104702/dybsvn.tar.gz
	-rw-r--r--  1 blyth blyth 1.5G May 18 14:01 /data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/05/18/104702/dybsvn.tar.gz
	-rw-r--r--  1 blyth blyth 1.5G May 19 14:02 /data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/05/19/104702/dybsvn.tar.gz
	-rw-r--r--  1 blyth blyth 7.3K May 17 13:53 /data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/05/17/104702/svnsetup.tar.gz
	-rw-r--r--  1 blyth blyth 7.3K May 18 13:38 /data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/05/18/104702/svnsetup.tar.gz
	-rw-r--r--  1 blyth blyth 7.3K May 19 13:39 /data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/05/19/104702/svnsetup.tar.gz


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

