
TODO
=====

Server Recreation 
------------------

Backups
^^^^^^^^^

#. cms02 backups 
#. add more backup nodes, 3 is not enough: belle1 + ? 
#. make dna mismatches get reported more loudly, or will they appear in the notification mail ?


Repository Migration to shared services ?
------------------------------------------

 * investigate moving more to github, **especially env** 
 * aberdeen repository is fat : and cannot be open source ?  
 * http://en.wikipedia.org/wiki/Comparison_of_open_source_software_hosting_facilities

Reboot Behaviour
-----------------

 * auto startup on reboot for hfag/cms01/cms02/belle7/belle1

    * hfag starts too much
    * cms01 fails to start "sv-blyth" 
    * cms02 not yet hooked up


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





