TODO
=====

Dayabay
--------

AdLidSensor Scraper
^^^^^^^^^^^^^^^^^^^^

#. hook up nosetests monitoring, from Christine

Backups
^^^^^^^

#. check on Shandong backups
#. check parasitic backups 
#. check offline_db backups

Sys Admin
-----------

Hang over from Yet Another NTU Powercut,  Thu 10 May 2012 ~13:30
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. cms01 reboot fails to start "sv-blyth" (wrong python message)
#. cms01 iptables setting was not presisted::
    
      [blyth@cms01 ~]$ iptables- ; IPTABLES_PORT=9090 iptables-webopen-ip $(local-tag2ip G)

#. cms02: httpd not auto started, and no error messages in /var/log/messages ... needs some further chkconfig magic ? manual start::

        [root@cms02 log]# /sbin/service httpd start


#. hfag: auto ntpupdate not working?  ~60 min behind.
#. hfag: reboot starts tomcat : must disable the chkconfig


cms02 Lockdown
^^^^^^^^^^^^^^^

#. move to ssh keyed access only for replacement cms02

Rsync Timeouts to N and N1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interactive scm-backup-rsync suffering timeouts whereas
not in the log from cron /var/scm/log/scm-backup-nightly.log 

#. seems to be network issue, cannot ping belle7 OR belle1 by name of number 
#. look for alternate backup targets... use mars


Exported Working Copies
^^^^^^^^^^^^^^^^^^^^^^^

#. bring work done during server outtage (mostly exported env) in from the cold : on belle1 + ? 

NUU network still preventing this::

        [blyth@belle1 ~]$ mv env env.b1
        [blyth@belle1 ~]$ svn co http://dayabay.phys.ntu.edu.tw/repos/env/trunk env
        svn: OPTIONS of 'http://dayabay.phys.ntu.edu.tw/repos/env/trunk': could not connect to server (http://dayabay.phys.ntu.edu.tw)
        [blyth@belle1 ~]$ 
        [blyth@belle1 ~]$ ping dayabay.phys.ntu.edu.tw
        PING cms02.phys.ntu.edu.tw (140.112.101.191) 56(84) bytes of data.

        --- cms02.phys.ntu.edu.tw ping statistics ---
        43 packets transmitted, 0 received, 100% packet loss, time 42000ms

Backups
^^^^^^^^^

#. manual backup checking : beyond operational basics
#. make dna mismatches get reported more loudly
#. cms02 backups are owned by **blyth** : lock em to prevent accidents ? 



Docs 
-----

#. NO NEED : DO THIS AS EDITING ANYHOW : svn postcommit hook to autorun the sphinx docs Makefile following commits into docs 
#. reposition sphinx control at top level allowing rst inclusion from anywhere in repo without symbolic links

   #. this would allow integration of bash ``precursor-usage`` into sphinx docs 

Repository Migration to shared services ?
------------------------------------------

 * investigate moving more to github, especially **env**, **tracdev**  
 * aberdeen repository is fat : and cannot be open source ?  
 * http://en.wikipedia.org/wiki/Comparison_of_open_source_software_hosting_facilities




