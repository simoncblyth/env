TODO
=====

heprez
-------

#. add new users for Yasmine CDF and D0 updating + open for editing

   #. http://dayabay.phys.ntu.edu.tw/tracs/heprez/ticket/104


env / scm-backup
------------------

#. move to full nodename for /var/scm/backup/NODENAME
#. hub based monitoring is convenient to check on the spokes, but means do not get notified when hub goes catatonic


Dayabay
--------

AdLidSensor Scraper
^^^^^^^^^^^^^^^^^^^^

#. hook up nosetests monitoring, from Christine

Backups
^^^^^^^

#. find other node for backups, as cms02 to belleX is likely to take awhile to fix
#. check on Shandong backups
#. check parasitic backups 
#. check offline_db backups

Env
----

Sys Admin
-----------

SCM 
^^^^

Hang over from Yet Another NTU Powercut,  Thu 10 May 2012 ~13:30
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. cms01 reboot fails to start "sv-blyth" (wrong python message)

   #. trouble is what to do depends on point in b2c work cycle

#. cms01 iptables setting was not presisted::
    
      [blyth@cms01 ~]$ iptables- ; IPTABLES_PORT=9090 iptables-webopen-ip $(local-tag2ip G)

#. cms02: httpd not auto started, and no error messages in /var/log/messages ... needs some further chkconfig magic ? manual start::

      [root@cms02 log]# /sbin/service httpd start

#. hfag: auto ntpupdate not working?  ~60 min behind.
#. hfag: reboot starts tomcat : must disable the chkconfig

cms02 Lockdown
^^^^^^^^^^^^^^^

#. move to ssh keyed access only for replacement cms02

   #. AWAITING NORMAL NETWORK RESUMPTION


NUU Network
^^^^^^^^^^^^

::

        [blyth@belle7 env]$ svn up
        svn: OPTIONS of 'http://dayabay.phys.ntu.edu.tw/repos/env/trunk': could not connect to server (http://dayabay.phys.ntu.edu.tw)
        [blyth@belle7 env]$ 

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

#. trac rst preview of sphinx flavored rst, has some errors due to unrecognized directived

   #. http://dayabay.phys.ntu.edu.tw/tracs/heprez/browser/trunk/log/end_of_2011.rst  **can trac be educated a bit for the most common ones**

#. NO NEED : DO THIS AS EDITING ANYHOW : svn postcommit hook to autorun the sphinx docs Makefile following commits into docs 
#. reposition sphinx control at top level allowing rst inclusion from anywhere in repo without symbolic links

   #. this would allow integration of bash ``precursor-usage`` into sphinx docs 

Repository Migration to shared services ?
------------------------------------------

 * investigate moving more to github, especially **env**, **tracdev**  
 * aberdeen repository is fat : and cannot be open source ?  
 * http://en.wikipedia.org/wiki/Comparison_of_open_source_software_hosting_facilities
 * https://bitbucket.org/  offers unlimited git or hg public and private repos, free for up to 5 users




