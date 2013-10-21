Migration steps
================

.. contents:: :local:

Gain access to target
-----------------------

::

    g4pb:renderer blyth$ ssh II
    Last login: Mon Oct 21 09:48:27 2013 from 10.10.5.168
    **********************************************************************
    |  Time  |      Up Time     |Loing Users|        Load Average        |
     09:55:14 up 90 days,  5:45, 13 users,  load average: 2.14, 1.61, 1.30
    **********************************************************************
    TEL:5037(office);83050656


    -bash-3.2$ ssh NN
    Last login: Mon Oct 21 09:56:59 2013 from lxslc507.ihep.ac.cn
    [root@dayabay1 ~]# 


checkout env SVN repo onto target 
-----------------------------------

::

    [root@dayabay1 ~]# pwd
    /root
    [root@dayabay1 ~]# svn co http://dayabay.phys.ntu.edu.tw/repos/env/trunk env


hookup env repo bash functions to bash shell
-----------------------------------------------

::

    [root@dayabay1 ~]# vi .bash_profile
    [root@dayabay1 ~]# tail -6 .bash_profile

    # hookup env bash functions
    export      ENV_HOME=$HOME/env       ; env-(){      [ -r $ENV_HOME/env.bash ]           && . $ENV_HOME/env.bash            && env-env $* ; }
    env-



