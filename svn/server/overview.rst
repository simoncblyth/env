SVN Server Overview
=====================

* http://svnbook.red-bean.com/en/1.7/svn.serverconfig.choosing.html



http, apache
-------------

Apache runs as nobody and writes into the repo::

    [blyth@cms02 ~]$ ll /var/scm/repos/newtest/
    total 72
    -rw-r--r--  1 nobody nobody  229 May  1  2012 README.txt
    drwxr-xr-x  2 nobody nobody 4096 May  1  2012 locks
    drwxr-xr-x  2 nobody nobody 4096 May  1  2012 hooks
    -r--r--r--  1 nobody nobody    2 May  1  2012 format
    drwxr-xr-x  2 nobody nobody 4096 May  1  2012 dav
    drwxr-xr-x  2 nobody nobody 4096 May  1  2012 conf
    drwxr-xr-x  7 nobody nobody 4096 May  1  2012 .
    drwxr-xr-x  8 nobody nobody 4096 May  7  2012 ..
    drwxr-sr-x  5 nobody nobody 4096 Jul 30  2012 db


svn+ssh, svnserve
-------------------

This can be configured to work in parallel with access via apache. 

  * http://svnbook.red-bean.com/nightly/en/svn.serverconfig.multimethod.html

It works by `ssh` starting a temporary `svnserve -t` which runs as the user invoking the command. 
Thus lots of care needed to avoid permissions problems.

Advantages:

#. potentially faster than http as more stateful
#. historically C2 http has been unexplicably blocked for months at a time from N, this would provide a workaround

   * but looking into the issues, suspect that tunnelling http over ssh would be easier than diddling with temporary svnserve


svn+ssh trial
---------------

.. warning:: CAUTION potential to mess up repo permissions, restrict to test repos

::

    [blyth@belle7 ~]$ svn co svn+ssh://blyth@dayabay.phys.ntu.edu.tw/var/scm/repos/newtest       
    svn: Expected version '3' of repository; found version '5'


Adjusting PATH and LD_LIBRARY_PATH in `C2:.cshrc` did not resolve the reported incompatibility
but it did change the svnserve version presumably being used::

    [blyth@belle7 ~]$ ssh C2 "tcsh -c \"which svnserve\""
    /usr/bin/svnserve
    [blyth@belle7 ~]$ ssh C2 "tcsh -c \"which svnserve\""
    /data/env/system/svn/subversion-1.4.6/bin/svnserve

Kinda difficult to debug as its a temporary tunnel.







