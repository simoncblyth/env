Installing Trac to virtualenv Python
====================================

.. contents:: :local:

Prerequisite: virtualenv 
------------------------

Install virtualenv to system Python::

        pip install virtualenv

Create an virtualenv environment for Trac::

        [webadmin@dayabay ~]$ mkdir ve
        [webadmin@dayabay ~]$ cd ve
        [webadmin@dayabay ve]$ virtualenv --no-site-packages tracs
        New python executable in tracs/bin/python
        Installing setuptools, pip...done.
        [webadmin@dayabay ve]$ ll tracs/
        total 20
        drwxrwxr-x. 3 webadmin webadmin 4096 Mar 28 12:08 ..
        drwxrwxr-x. 3 webadmin webadmin 4096 Mar 28 12:08 lib
        drwxrwxr-x. 2 webadmin webadmin 4096 Mar 28 12:08 include
        drwxrwxr-x. 5 webadmin webadmin 4096 Mar 28 12:08 .
        drwxrwxr-x. 2 webadmin webadmin 4096 Mar 28 12:08 bin
        [webadmin@dayabay ve]$ 

Activate the virtualenv environment::

        [webadmin@dayabay ve]$ cd tracs/
        [webadmin@dayabay tracs]$ . bin/activate
        (tracs)[webadmin@dayabay tracs]$ 

Note the change in the command prompt after the activation.

Install Trac with tracbuild-auto()
----------------------------------

Make sure that the virtualenv environment has already been activated. Then install Trac and relevant packages with::

        (tracs)[webadmin@dayabay tracs]$ trac-
        (tracs)[webadmin@dayabay tracs]$ tracbuild-
        (tracs)[webadmin@dayabay tracs]$ tracbuild-auto

Check the installation::

        (tracs)[webadmin@dayabay tracs]$ ll lib/python2.6/site-packages/
        total 300
        drwxrwxr-x.  4 webadmin webadmin   4096 Mar 28 12:08 ..
        -rw-rw-r--.  1 webadmin webadmin    126 Mar 28 12:08 easy_install.py
        -rw-rw-r--.  1 webadmin webadmin  99605 Mar 28 12:08 pkg_resources.py
        -rw-rw-r--.  1 webadmin webadmin    325 Mar 28 12:08 easy_install.pyc
        -rw-rw-r--.  1 webadmin webadmin 108735 Mar 28 12:08 pkg_resources.pyc
        drwxrwxr-x.  4 webadmin webadmin   4096 Mar 28 12:08 setuptools
        drwxrwxr-x.  2 webadmin webadmin   4096 Mar 28 12:08 _markerlib
        drwxrwxr-x.  2 webadmin webadmin   4096 Mar 28 12:08 setuptools-2.2.dist-info
        drwxrwxr-x.  6 webadmin webadmin   4096 Mar 28 12:08 pip
        drwxrwxr-x.  2 webadmin webadmin   4096 Mar 28 12:08 pip-1.5.4.dist-info
        drwxrwxr-x.  4 webadmin webadmin   4096 Apr  1 15:45 Genshi-0.5-py2.6-linux-i686.egg
        drwxrwxr-x.  4 webadmin webadmin   4096 Apr  1 15:46 Trac-0.11-py2.6.egg
        drwxrwxr-x.  4 webadmin webadmin   4096 Apr  1 15:46 Bitten-0.6dev_r561-py2.6.egg
        drwxrwxr-x.  4 webadmin webadmin   4096 Apr  1 15:46 TracAccountManager-0.2.1dev_r3857-py2.6.egg
        drwxrwxr-x.  4 webadmin webadmin   4096 Apr  1 15:46 BitExtra-0.0.1-py2.6.egg
        drwxrwxr-x.  4 webadmin webadmin   4096 Apr  1 15:46 BittenNotify-0.1dev_r28-py2.6.egg
        drwxrwxr-x.  4 webadmin webadmin   4096 Apr  1 15:46 NavAdd-0.1-py2.6.egg
        drwxrwxr-x.  4 webadmin webadmin   4096 Apr  1 15:46 Pygments-1.6-py2.6.egg
        drwxrwxr-x.  5 webadmin webadmin   4096 Apr  1 15:47 SvnAuthzAdminPlugin-0.1.2._Moved.to.Trac.0.11_-py2.6.egg
        drwxrwxr-x.  3 webadmin webadmin   4096 Apr  1 15:47 textile-2.0.11-py2.6.egg
        drwxrwxr-x.  4 webadmin webadmin   4096 Apr  1 15:47 TracNav-4.0pre6-py2.6.egg
        drwxrwxr-x.  4 webadmin webadmin   4096 Apr  1 15:47 TracTags-0.6-py2.6.egg
        -rw-rw-r--.  1 webadmin webadmin    602 Apr  1 15:47 easy-install.pth
        drwxrwxr-x. 20 webadmin webadmin   4096 Apr  1 15:47 .
        drwxrwxr-x.  4 webadmin webadmin   4096 Apr  1 15:47 TracTocMacro-11.0.0.3-py2.6.egg
        (tracs)[webadmin@dayabay tracs]$ ll bin/
        total 68
        lrwxrwxrwx. 1 webadmin webadmin    6 Mar 28 12:08 python2 -> python
        -rwxrwxr-x. 1 webadmin webadmin 6088 Mar 28 12:08 python
        lrwxrwxrwx. 1 webadmin webadmin    6 Mar 28 12:08 python2.6 -> python
        -rwxrwxr-x. 1 webadmin webadmin  251 Mar 28 12:08 easy_install-2.6
        -rwxrwxr-x. 1 webadmin webadmin  251 Mar 28 12:08 easy_install
        -rwxrwxr-x. 1 webadmin webadmin  223 Mar 28 12:08 pip2.6
        -rwxrwxr-x. 1 webadmin webadmin  223 Mar 28 12:08 pip2
        -rwxrwxr-x. 1 webadmin webadmin  223 Mar 28 12:08 pip
        -rw-rw-r--. 1 webadmin webadmin 1129 Mar 28 12:08 activate_this.py
        -rw-rw-r--. 1 webadmin webadmin 2473 Mar 28 12:08 activate.fish
        -rw-rw-r--. 1 webadmin webadmin 1260 Mar 28 12:08 activate.csh
        -rw-rw-r--. 1 webadmin webadmin 2204 Mar 28 12:08 activate
        -rwxrwxr-x. 1 webadmin webadmin  304 Apr  1 15:46 tracd
        -rwxrwxr-x. 1 webadmin webadmin  314 Apr  1 15:46 trac-admin
        -rwxrwxr-x. 1 webadmin webadmin  345 Apr  1 15:46 bitten-slave
        -rwxrwxr-x. 1 webadmin webadmin  323 Apr  1 15:46 pygmentize
        drwxrwxr-x. 2 webadmin webadmin 4096 Apr  1 15:46 .
        drwxrwxr-x. 5 webadmin webadmin 4096 Apr  1 15:53 ..
        (tracs)[webadmin@dayabay tracs]$ 

Setup alternative frontend
--------------------------

Follow the instruction in http://trac.edgewall.org/wiki/TracDev/AlternativeFrontends to setup an alternative frontend. In particular, edit ``/etc/httpd/conf/svnsetup/tracs.conf`` and change::

        PythonPath "sys.path + ['/usr/lib/python2.6/site-packages']"
        PythonHandler trac.web.modpython_frontend 

to::

        PythonPath "['/home/webadmin/ve/tracs/bin'] + sys.path"
        PythonHandler virtualtrac

And create a file ``virtualtrac.py`` under ``/home/webadmin/ve/tracs/bin/`` with the following contents::

        import os
        import site
        site.addsitedir('/home/webadmin/ve/tracs/lib/python2.6/site-packages')

        from trac.web.modpython_frontend import handler

Then start/restart Apache.
