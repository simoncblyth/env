SOP : for servers
====================

general
--------

::

    [blyth@cms01 env]$ sudo /sbin/shutdown now
    Broadcast message from root (pts/0) (Sat Mar 23 15:49:21 2013):
    The system is going down to maintenance mode NOW!


Its taking a long while to stop though::

    blyth     7679  0.0  0.0  8960 1792 ?        S    13:22   0:00 sshd: blyth@pts/0
    blyth     7680  0.0  0.1  7336 3576 pts/0    Ss   13:22   0:07 -bash
    root     14566  0.0  0.0  5984 1324 ?        Ss   15:49   0:00 /bin/bash /etc/rc.d/rc 1
    root     14690  0.0  0.0  4460  740 ?        S    15:49   0:00 initlog -q -c /etc/rc1.d/K20sv-blyth stop
    root     14691  0.0  0.0  4832 1272 ?        S    15:49   0:00 /bin/sh /etc/rc1.d/K20sv-blyth stop
    root     14697  0.2  0.3 11716 6440 ?        S    15:49   0:00 /data/env/system/python/Python-2.5.1/bin/python /data/env/system/python/Python-2.5.1/bin/supervisorctl -c /data/env/local/env/sv/ctl/C.ini shutdown
    blyth    14705  0.0  0.0  3480  780 pts/0    R+   15:51   0:00 ps aux
    root 


hfag
----

::

    [blyth@hfag blyth]$ sudo /sbin/service tomcat status
    Password:
    Tomcat is stopped...
    [blyth@hfag blyth]$ sudo /sbin/service tomcat status
    Tomcat is stopped...

    [blyth@hfag blyth]$ sudo /sbin/service exist status
    eXist Native XML Database is running (1771).
    [blyth@hfag blyth]$ sudo /sbin/service exist stop
    Stopping eXist Native XML Database...
    Stopped eXist Native XML Database.
    [blyth@hfag blyth]$ 


It takes a little while to issue all the stops::

    [blyth@hfag blyth]$ sudo /sbin/shutdown now

    Broadcast message from root (pts/0) (Sat Mar 23 14:58:28 2013):

    The system is going down to maintenance mode NOW!

    [blyth@hfag blyth]$ ps aux
    USER       PID %CPU %MEM   VSZ  RSS TTY      STAT START   TIME COMMAND
    root         1  0.0  0.1  1516  504 ?        S     2012   0:10 init
    root         2  0.0  0.0     0    0 ?        SW    2012   0:00 [keventd]
    root         3  0.0  0.0     0    0 ?        SW    2012   0:00 [kapmd]
    root         4  0.0  0.0     0    0 ?        SWN   2012   0:00 [ksoftirqd/0]
    root         7  0.0  0.0     0    0 ?        SW    2012   0:00 [bdflush]
    root         5  0.0  0.0     0    0 ?        SW    2012  16:20 [kswapd]
    root         6  0.0  0.0     0    0 ?        SW    2012  24:29 [kscand]
    root         8  0.0  0.0     0    0 ?        SW    2012   0:27 [kupdated]
    root         9  0.0  0.0     0    0 ?        SW    2012   0:00 [mdrecoveryd]
    root        13  0.0  0.0     0    0 ?        SW    2012   0:03 [kjournald]
    root        68  0.0  0.0     0    0 ?        SW    2012   0:00 [khubd]
    root      1041  0.0  0.0     0    0 ?        SW    2012   0:00 [kjournald]
    root      1042  0.0  0.0     0    0 ?        SW    2012   3:49 [kjournald]
    root      1043  0.0  0.0     0    0 ?        SW    2012   0:32 [kjournald]
    root      1433  0.0  0.2  1580  548 ?        S     2012   0:06 syslogd -m 0
    root      1437  0.0  0.1  1524  460 ?        S     2012   0:00 klogd -x
    root       834  0.0  0.7  6856 1880 ?        S    14:52   0:00 sshd: blyth [priv]
    blyth      836  0.0  0.8  7012 2132 ?        S    14:52   0:00 sshd: blyth@pts/0
    blyth      838  0.1  0.9  6416 2432 pts/0    S    14:52   0:00 -bash
    root      1740  0.3  0.5  5344 1288 ?        S    14:58   0:00 /bin/bash /etc/rc.d/rc 1
    root      1980  0.0  0.5  5344 1280 ?        S    14:58   0:00 /bin/bash /etc/rc1.d/K88syslog stop
    root      1984  0.0  0.2  4932  560 ?        S    14:58   0:00 sleep 1
    blyth     1985  0.0  0.2  2732  700 pts/0    R    14:58   0:00 ps aux
    [blyth@hfag blyth]$ 



hfag restart requires manual F2 keypress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




hfag nginx reverse proxy 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used as reverse proxy for SVN on C2, allowing NUU nodes that are routinely 
blocked from accessing node C2 apache (for SVN) to have access to the repositories. 
The NUU blocks are assumed to be due to C2s habit of transferring 
gigabytes per day of backup tarballs. 

*Not auto-started on reboot*, to start::

    nginx-
    nginx-sstart

    [blyth@hfag blyth]$ t nginx-sstart
    nginx-sstart is a function
    nginx-sstart () 
    { 
        $SUDO nginx
    }


Without this get::

    [blyth@belle7 e]$ svn up
    svn: OPTIONS of 'http://hfag.phys.ntu.edu.tw:90/repos/env/trunk': could not connect to server (http://hfag.phys.ntu.edu.tw:90)



cms02  : as repo server this one goes last
--------------------------------------------

Hmm, considered doing an extra backup. But have stopped two of the targets
already. Anyhow working copy is uptodate on G, so would not loose significantly
if needed to restore from yesterdays backup.


cms02 : restarting scm backup system after a reboot
------------------------------------------------------

check/remove LOCKED backup dirs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do this first, as tends to be otherwise forgotten causing subsequent backup failures::

    [root@cms02 ~]# cd /var/scm
    [root@cms02 scm]# ls -l 
    total 64
    drwxr-xr-x   4 blyth  blyth  4096 Sep 18  2012 backup
    drwxr-xr-x   2 nobody nobody 4096 Mar 19 12:45 conf
    drwxr-xr-x   3 blyth  blyth  4096 May  8  2012 foreign
    drwxr-xr-x   2 root   root   4096 Apr  1 21:19 LOCKED
    drwxr-xr-x   2 root   root   4096 Apr  1 15:50 log
    drwxr-xr-x   8 nobody nobody 4096 May  7  2012 repos
    drwxr-xr-x   2 nobody nobody 4096 May  4  2012 svn
    drwxr-xr-x  11 nobody nobody 4096 May  7  2012 tracs
    [root@cms02 scm]# ll LOCKED
    total 4
    -rw-r--r--  1 root root 0 Apr  1 21:19 scm-backup-rsync-started-2013-04-01@21:19:28
    [root@cms02 scm]# date
    Tue Apr  2 12:11:21 CST 2013
    [root@cms02 scm]# rm -rf LOCKED
    [root@cms02 scm]# 



restore ssh agent
~~~~~~~~~~~~~~~~~~

::

    simon:~ blyth$ ssh C2R
    Last login: Tue Mar 12 20:47:07 2013 from simon.phys.ntu.edu.tw
    [root@cms02 ~]# . .bash_profile 
    [root@cms02 ~]# sas
    ===== sourcing the info for the agent /root/.ssh-agent-info-C2R
    ===== adding identities to the agent
    [root@cms02 ~]# exit


check/edit crontab times
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [root@cms02 ~]# crontab -l
    SHELL = /bin/bash
    15 21 * * *  ( export HOME=/root ; export NODE=cms02 ; export MAILTO=blyth@hep1.phys.ntu.edu.tw ; export ENV_HOME=/home/blyth/env ; . /home/blyth/env/env.bash ; env-  ; scm-backup- ; scm-backup-nightly ) >  /var/scm/log/scm-backup-nightly-$(date +"\%d").log 2>&1
    15 22 * * *  ( export HOME=/root ; export NODE=cms02 ; export MAILTO=blyth@hep1.phys.ntu.edu.tw ; export ENV_HOME=/home/blyth/env ; . /home/blyth/env/env.bash ; env-  ; scm-backup- ; scm-backup-tgzmon ) >  /var/scm/log/scm-backup-tgzmon-$(date +"\%d").log 2>&1
    50 * * * * ( export HOME=/root ; LD_LIBRARY_PATH=/data/env/system/python/Python-2.5.6/lib /data/env/system/python/Python-2.5.6/bin/python /home/blyth/env/db/valmon.py -s oomon rec ; ) > /var/scm/log/oomon.log 2>&1





cms01 
-----

heprez servers : exist httpd tomcat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

supervisord and contained mysql
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 cronlog]$ sv
    mysql                            RUNNING    pid 9604, uptime 191 days, 21:08:16
    C> stop mysql
    mysql: stopped

    C> shutdown
    Really shut the remote supervisord process down y/N? y
    Shut down
    C> 
    [blyth@cms01 cronlog]$ 


rabbitmq-server
~~~~~~~~~~~~~~~

::

    [blyth@cms01 env]$ sudo /sbin/service rabbitmq-server status
    Status of all running nodes...
    Node 'rabbit@cms01' with Pid 3131: running
    done.
    [blyth@cms01 env]$ sudo /sbin/service rabbitmq-server stop
    Stopping rabbitmq-server: rabbitmq-server.
    [blyth@cms01 env]$ 
    [blyth@cms01 env]$ ll /etc/init.d/


xinetd
~~~~~~~

::

    [blyth@cms01 env]$ sudo /sbin/service xinetd stop
    Stopping xinetd:                                           [  OK  ]


