Installing **env** 
======================

**env** bash functions 
---------------------------

The env bash functions are installed by 

#. checkout env from SVN::

    [root@belle1 ~]# cd 
    [root@belle1 ~]# pwd
    /root
    [root@belle1 ~]# svn co http://dayabay.phys.ntu.edu.tw/repos/env/trunk env

#. hook up the bash functions to your bash shell by adding the below to your :file:`.bash_profile`::

    export ENV_HOME=/root/env ; env-(){ [ -r $ENV_HOME/env.bash ] && . $ENV_HOME/env.bash  && env-env $* ; }
    env-
    PATH=$ENV_HOME/bin:$PATH
    export PATH


**env** python modules
------------------------

#. find the system python site packages, eg at `/usr/lib/python2.4/site-packages`::

    [root@belle1 tmp]# python -c "import sys ; print '\n'.join(sys.path) "     # finding site-packages

    /usr/lib/python24.zip
    /usr/lib/python2.4
    /usr/lib/python2.4/plat-linux2
    /usr/lib/python2.4/lib-tk
    /usr/lib/python2.4/lib-dynload
    /usr/lib/python2.4/site-packages
    /usr/lib/python2.4/site-packages/Numeric
    /usr/lib/python2.4/site-packages/gtk-2.0

#. plant symbolic link to env python modules::

    [root@belle1 tmp]# cd /usr/lib/python2.4/site-packages
    [root@belle1 site-packages]# ln -s /root/env env 
    [root@belle1 site-packages]# ll env*
    lrwxrwxrwx 1 root root 15 May 30 18:28 env -> /root/env

#. check can access the `env` python modules::

    [root@belle1 ~]# python -c "import env"
    [root@belle1 ~]# 


example crontab using **env** python scripts and modules
---------------------------------------------------------

Edit crontab with `crontab -e`::

    SHELL=/bin/bash
    HOME=/root
    ENV_HOME=/root/env
    CRONLOG_DIR=/root/cronlog
    MAILTO=blyth@hep1.phys.ntu.edu.tw
    PATH=/root/env/bin:/usr/bin:/bin
    #
    40 05 * * * ( . $ENV_HOME/env.bash ; db- ; db-backup-rsync-monitor )  > $CRONLOG_DIR/db-backup-rsync-monitor.log 2>&1
    52 * * * * ( valmon.py -s diskmon rec rep mon )                       > $CRONLOG_DIR/diskmon.log 2>&1 

For more on the `valmon.py` script see :doc:`/db/valmon`.

