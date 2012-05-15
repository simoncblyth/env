
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

cms01 httpd start error
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

        [blyth@cms01 logs]$ heprez-svc start tomcat httpd
        ...
        === heprez-svc : sleeping for 5 before start httpd
        Starting httpd: Syntax error on line 15 of /etc/httpd/conf/svnsetup/tracs.conf:
        Invalid command 'PythonHandler', perhaps mis-spelled or defined by a module not included in the server configuration
        [FAILED]

        
comment the ``apache-edit`` left over from cms02 search for stand-in

::

        #Include /etc/httpd/conf/svnsetup/setup.conf 


Succeeded to preview and edit, **cms01 survived powercut unscathed**

* http://cms01.phys.ntu.edu.tw/rezdb/db/test/


Yet Another NTU Powercut,  Thu 10 May 2012 ~13:30
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. cms01 : cannot access cms01, no ping::

        simon:env blyth$ date
        Thu 10 May 2012 14:17:36 CST
        simon:env blyth$ ping cms01.phys.ntu.edu.tw
        PING cms01.phys.ntu.edu.tw (140.112.101.190): 56 data bytes

     #. from console, twas stuck at BIOS initialization ... powercycling regained access

     #. usual manual mount:: 
     
           [blyth@cms01 ~]$ sudo mount /data  

     #. do a manual ``exist-start`` as improper shutdown, this hangs ... but doing a exist-service-start succeeds
        and the XMLDB is operational, succeeded to to a heprez-propagate to G for backup

     #. this iptables setting was not presisted::
    
            [blyth@cms01 ~]$ iptables- ; IPTABLES_PORT=9090 iptables-webopen-ip $(local-tag2ip G)


#. cms02 :  httpd was not auto started, and no error messages in /var/log/messages ... needs some further chkconfig magic ?, manual start succeeded::

        [root@cms02 log]# /sbin/service httpd start

#. hfag : again started too much, manually stop tomcat and exist, httpd OK::

        [blyth@hfag blyth]$ sudo /sbin/service tomcat stop
        [blyth@hfag blyth]$ sudo /sbin/service exist stop



hfag auto ntp not working
^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that hfag is ~60 min behind.


Rsync Timeouts to N and N1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interactive scm-backup-rsync suffering timeouts whereas
not in the log from cron /var/scm/log/scm-backup-nightly.log 

#. seems to be network issue, cannot ping belle7 OR belle1 by name of number 
#. look for alternate backup targets... use mars




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

#. improve the ``env-index`` : 

   #. **DEDICATE A SPHINX-STANCE TO HOLDING THE LINKS** easier .rst source management

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





