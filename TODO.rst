
TODO
=====

Dayabay
--------

Backups
^^^^^^^

#. check on Shandong backups
#. check parasitic backups 
#. check offline_db backups

Server Rebuild 
----------------

Lockdown
^^^^^^^^^

#. move to ssh keyed access only 

Exported Working Copies
^^^^^^^^^^^^^^^^^^^^^^^

#. bring work done during server outtage (mostly exported env) in from the cold : on belle1 + ? 

Backups
^^^^^^^^^

#. manual backup checking : beyond operational basics
#. make dna mismatches get reported more loudly
#. backups are owned by **blyth** : lock em to prevent accidents ? 

Docs 
-----

#. improve the ``env-index``
#. svn postcommit hook to autorun the sphinx docs Makefile following commits into docs 
#. dated and revisioned docs in index.rst : can this be done without resort to templates 

Repository Migration to shared services ?
------------------------------------------

 * investigate moving more to github, especially **env**, **tracdev**  
 * aberdeen repository is fat : and cannot be open source ?  
 * http://en.wikipedia.org/wiki/Comparison_of_open_source_software_hosting_facilities

Reboot Behaviour
-----------------

 * auto startup on reboot for hfag/cms01/cms02/belle7/belle1

    * hfag starts too much
    * cms01 fails to start "sv-blyth" 
    * cms02 : httpd is now added to chkconfig 


hfag chkconfig : starts undesired services
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

        [blyth@hfag blyth]$ ls -l /etc/init.d/ | grep /data
        lrwxrwxrwx    1 root     root           51 May 16  2008 apache -> /data/usr/local/apache2/httpd-2.0.59/sbin/apachectl
        lrwxrwxrwx    1 root     root           51 May  7  2007 apache2 -> /data/usr/local/apache2/httpd-2.0.59/sbin/apachectl

        #!/bin/sh
        # chkconfig: 345 50 50 
        # description: apachectl

        lrwxrwxrwx    1 root     root           96 May 16  2008 exist -> /data/usr/local/heprez/install/exist/eXist-snapshot-20051026/unpack/4/tools/wrapper/bin/exist.sh

           no chkconfig setup

        lrwxrwxrwx    1 root     root          103 May 16  2008 tomcat -> /data/usr/local/heprez/install/tomcat/jakarta-tomcat-4.1.31/2/jakarta-tomcat-4.1.31/../../etc/tomcat.sh

        # chkconfig: 345 91 10
        # description: Starts and stops the Tomcat daemon.
        #





