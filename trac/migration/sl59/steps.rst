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


get into env
--------------

::

    simon:e blyth$ env-   # if not already done in .bash_profile
    simon:e blyth$ t t 
    t is aliased to `type'


env bash function usage guidelines
------------------------------------

#. Update SVN working copy and pickup changes with `env-` etc.. before using the bash functions
#. Use the **t** alias to follow along what unfamiliar functions are doing before 
   using them, in order to ensure no harm will result and to allow easier fixing when they fail 
#. Commit changes to the functions to the env SVN repository, **with a brief but informative commit message**


node characterisation
-----------------------

The `env` seeks to factor away node specifics, via definition of 
a **NODE_TAG** for each node and bash functions with echoing case statements 
for standard things eg::

    [root@dayabay1 e]# t local-var-base
    local-var-base is a function
    local-var-base () 
    { 
        local t=${1:-$NODE_TAG};
        case $t in 
            U)
                echo /var
            ;;
            P)
                echo /disk/d3/var
            ;;
            G1)
                echo /disk/d3/var
     ...
    }

::

    [root@dayabay1 e]# local-info

       For tag Y1   (actual node is Y1) 

       local-server-tag  : P   node designated as the source node holding the repository
       local-restore-tag : H1  node holding the backup tarballs of the designated server node 
       local-backup-tag  : U   paired node to which backups are sent from Y1  

       local-system-base :  /usr/local
       local-base        :  /usr/local
       local-var-base    :  /var
       local-base-env    :  /usr/local/env
       local-scm-fold    :  /home/scm
       local-user-base   :  /tmp
       local-output-base :  /tmp


characterizing a new node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. pick an unused NODE_TAG (eg Y1 for dayabay1) see `local-nodetag`
#. extend case statements to accomodate the new tag, see `local-info`


trac build and instance migration
-----------------------------------

The primary entry point bash functions are:

`trac-build`
       local node initialization, 
       prerequiste installs appropriate to the configured mode (system OR source),
       getting and installing ~15 python packages, including trac, bitten, bittennotify, ... 

`svnsetup-sysapache`
       writes apache config files for Trac/SVN to be included into httpd.conf

`scm-recover-all`
       expands the backup tarballs for the Trac and SVN instances and does configurations


`trac-build`
~~~~~~~~~~~~~~~
    
::


    simon:migration blyth$ t trac-build  
    trac-build is a function
    trac-build () 
    { 
        local-;
        local-initialize;
        tracpreq-;
        tracpreq-again;
        tracbuild-;
        tracbuild-auto
    }


Trac pre-requisites are obtained and build by `tracpreq-again`. This operates in source and system modes.
In source mode the sources for SVN/apache/swig/python/... are downloaded and build, whereas in 
system mode only two python packages are grabbed: setuptools, configobj 

::

    simon:e blyth$ t tracpreq-mode-default
    tracpreq-mode-default is a function
    tracpreq-mode-default () 
    { 
        case ${1:-$NODE_TAG} in 
            ZZ | C | Y1)
                echo system
            ;;
            *)
                echo source
            ;;
        esac
    }



`tracbuild-auto` gets/installs the packages listed by `tracbuild-names`::

    [root@dayabay1 e]# trac-
    [root@dayabay1 e]# tracbuild-
    [root@dayabay1 e]# tracbuild-names
    genshi tractrac bitten accountmanager bittennotify fullblog navadd pygments silvercity svnauthzadmin textile tracdoxygen tracnav tractags tractoc


The packages are mostly checked out from the original SVN repositories 

  * TODO: archive the packages and place the tarballs somewhere accessible to avoid dependencies on ~15 remote SVN servers
  * (Oct 2013, the bittennotify svn server was found to no longer be accessible)


interference with pre-existing packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-installation of any of these packages is liable to cause issues.  It is simplest to 
uninstall them if they are already present in the python being used.


test migration of dybsvn to dayabay1 (Oct 2013)
-------------------------------------------------

Initially perform staight copy of dayabay/dybsvn with no version changes or other improvements.

Problems encountered:

#. `configobj-build` had to be done manually, why not automated ?
#. `tracbuild-auto` ran 
#. Trac configuration stomps upon  "query,daily" changing to "query"

   * probably the "daily" was added manually, without inclusion into the functions


improvements
---------------

#. AccountManager plugin is outdated (has security issues), and needs to be updated
#. bitten build html formatting need to be made wider


