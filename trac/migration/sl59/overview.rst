Trac Migration to SL59
========================

review current instance installation
------------------------------------

System apache config pulls in config::

    Include /etc/httpd/conf/svnsetup/setup.conf

which then pulls::

    Include /etc/httpd/conf/svnsetup/repos.conf
    Include /etc/httpd/conf/svnsetup/svn.conf
    Include /etc/httpd/conf/svnsetup/tracs.conf

The `tracs.conf` connects to mod python in system python2.3::

    <Location /tracs>
       SetHandler mod_python
       PythonPath "sys.path + ['/usr/lib/python2.3/site-packages']"
       PythonHandler trac.web.modpython_frontend 
       PythonOption TracEnvParentDir /home/scm/tracs
       PythonOption TracUriRoot /tracs


current server
--------------------

::

    [dayabay] /home/blyth > uname -a
    Linux dayabay.ihep.ac.cn 2.6.9-104.ELsmp #1 SMP Fri Jul 6 09:14:38 CEST 2012 i686 i686 i386 GNU/Linux

    [dayabay] /home/blyth > cat /etc/redhat-release
    Scientific Linux CERN SLC release 4.8 (Beryllium)

    [dayabay] /home/blyth > python -V
    Python 2.3.4

    [dayabay] /home/blyth > python -c "import sys ; print '\n'.join(sys.path) "

    /usr/lib/python2.3/site-packages/configobj-4.6.0-py2.3.egg
    /usr/lib/python2.3/site-packages/Genshi-0.5-py2.3-linux-i686.egg
    /usr/lib/python2.3/site-packages/Trac-0.11-py2.3.egg
    /usr/lib/python2.3/site-packages/TracAccountManager-0.2.1dev_r3857-py2.3.egg
    /usr/lib/python2.3/site-packages/BittenNotify-0.1dev_r28-py2.3.egg
    /usr/lib/python2.3/site-packages/TracFullBlogPlugin-0.1-py2.3.egg
    /usr/lib/python2.3/site-packages/NavAdd-0.1-py2.3.egg
    /usr/lib/python2.3/site-packages/SilverCity-0.9.7-py2.3-linux-i686.egg
    /usr/lib/python2.3/site-packages/textile-2.0.11-py2.3.egg
    /usr/lib/python2.3/site-packages/Trac2Latex-0.0.1-py2.3.egg
    /usr/lib/python2.3/site-packages/Trac2MediaWiki-0.0.2-py2.3.egg
    /usr/lib/python2.3/site-packages/TracNav-4.0pre6-py2.3.egg
    /usr/lib/python2.3/site-packages/TracTags-0.6-py2.3.egg
    /usr/lib/python2.3/site-packages/TracTocMacro-11.0.0.3-py2.3.egg
    /usr/lib/python2.3/site-packages/setuptools-0.6c11-py2.3.egg
    /usr/lib/python2.3/site-packages/virtualenv-1.4.3.post1-py2.3.egg
    /usr/lib/python2.3/site-packages/pip-0.6.post1-py2.3.egg
    /usr/lib/python2.3/site-packages/supervisor-3.0a7-py2.3.egg
    /usr/lib/python2.3/site-packages/elementtree-1.2.7_20070827_preview-py2.3.egg
    /usr/lib/python2.3/site-packages/meld3-0.6.6-py2.3.egg
    /usr/lib/python2.3/site-packages/hashlib-20081119-py2.3-linux-i686.egg
    /usr/lib/python2.3/site-packages/Bitten-0.6dev_r561-py2.3.egg
    /usr/lib/python2.3/site-packages/BitExtra-0.0.1-py2.3.egg
    /usr/lib/python23.zip
    /usr/lib/python2.3
    /usr/lib/python2.3/plat-linux2
    /usr/lib/python2.3/lib-tk
    /usr/lib/python2.3/lib-dynload
    /usr/lib/python2.3/site-packages
    /usr/lib/python2.3/site-packages/gtk-2.0


    [dayabay] /home/blyth > svn --version
    svn, version 1.4.4 (r25188)
       compiled Nov 21 2007, 11:15:10



dayabay1 test machine
-----------------------

Double hop required to get in::

    simon:~ blyth$ ssh II
    Last login: Fri Sep 13 16:06:11 2013 from simon.phys.ntu.edu.tw
    **********************************************************************
    |  Time  |      Up Time     |Loing Users|        Load Average        |
     16:08:44 up 52 days, 11:59, 28 users,  load average: 1.19, 0.96, 1.07
    **********************************************************************
    TEL:5037(office);83050656

    -bash-3.2$ ssh NN
    Last login: Fri Sep 13 16:07:57 2013 from lxslc514.ihep.ac.cn

    [root@dayabay1 ~]# uname -a
    Linux dayabay1 2.6.18-308.el5 #1 SMP Fri Feb 24 08:42:10 CET 2012 x86_64 x86_64 x86_64 GNU/Linux

    [root@dayabay1 ~]# cat /etc/redhat-release 
    Scientific Linux CERN SLC release 5.9 (Boron)

    [root@dayabay1 ~]# python -V
    Python 2.4.3

    [root@dayabay1 ~]# python -c "import sys ; print '\n'.join(sys.path) "

    /usr/lib64/python24.zip
    /usr/lib64/python2.4
    /usr/lib64/python2.4/plat-linux2
    /usr/lib64/python2.4/lib-tk
    /usr/lib64/python2.4/lib-dynload
    /usr/lib64/python2.4/site-packages
    /usr/lib64/python2.4/site-packages/Numeric
    /usr/lib64/python2.4/site-packages/gtk-2.0
    /usr/lib/python2.4/site-packages

    [root@dayabay1 ~]# which svn
    /usr/bin/svn

    [root@dayabay1 ~]# svn --version
    svn, version 1.6.11 (r934486)
       compiled Apr 12 2013, 08:05:38


svn migrate 1.4 1.6
---------------------

* :google:`svn migrate 1.4 1.6`
* http://stackoverflow.com/questions/1435047/migrate-from-subversion-1-4-to-1-6

::

    svnadmin create --pre-1.5-compatible /path/to/repo
    svnadmin load /path/to/repo <dumpfile
    svnadmin upgrade /path/to/repo

svnbook on migration
~~~~~~~~~~~~~~~~~~~~~~~~

* http://svnbook.red-bean.com/nightly/en/svn.ref.svnadmin.html#svn.ref.svnadmin.c

`--pre-1.4-compatible`
           When creating a new repository, use a format that is compatible with versions of Subversion earlier than Subversion 1.4.
`--pre-1.5-compatible`
           When creating a new repository, use a format that is compatible with versions of Subversion earlier than Subversion 1.5.
`--pre-1.6-compatible`
           When creating a new repository, use a format that is compatible with versions of Subversion earlier than Subversion 1.6.


svn version migration
~~~~~~~~~~~~~~~~~~~~~~~

This looks forced as we have been using system 1.4 and the new system uses 1.6.


trac version migration 
----------------------

System (yum distro) versions are 0.10.4, 0.10.5 on the old and new. 
We have been using 0.11, no compelling reason to change.


Problematic due to bitten, the patches that have been applied, and the large number of plugins.

On old machine we use python2.3 site-packages level install, not system level with `/usr/lib/python2.3/site-packages/Trac-0.11-py2.3.egg`


version pinned due to bitten and patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bitten trac plugin provides a master/slave system that requires (from bitter experience) 
that the slaves (installed at BNL,LBNL,IHEP,NTU,NUU) use precisely 
the same version of bitten that the master Trac instance does.  This effectively pins the 
bitten version of the master, without having to reinstall all the slaves.

Other reasons to keep the same version:

  * Pinned revisions of ~15 plugins are known to work with the current version

       * http://dayabay.phys.ntu.edu.tw/repos/env/trunk/trac/package/  

  * ~8 patches to trac and plugins require precisely the same revisions to be used

       * http://dayabay.phys.ntu.edu.tw/repos/env/trunk/trac/patch/ 
 

what to change ?
~~~~~~~~~~~~~~~~~

As little as possible, but:

#. update AccountManager plugin
#. remember docutils and RST support in Trac
#. check the tickets and notes for long forgotten things to change, eg bitten build web UI


