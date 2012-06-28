TODO
=====

Dayabay
--------

AdLidSensor Scraper
^^^^^^^^^^^^^^^^^^^^

#. hook up nosetests monitoring, from Christine

Docs
^^^^^

#. bashtoctree


hep1
^^^^^

::

        g4pb-2:~ blyth$ ssh H1
        ssh_exchange_identification: Connection closed by remote host
        g4pb-2:~ blyth$ 



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

Improve SCM logging
^^^^^^^^^^^^^^^^^^^^^

::

   Unfortunately the log /var/log/env/scm-backup.log
   has overwritten the one with the original failure
   (I will look into improving logging to prevent that in future).

   Done on cms02, propagate to Q after port 22 opened 


Improve Monitoring
^^^^^^^^^^^^^^^^^^^

#. use fabric from a cron job to capture daily backup metrics (collected from various ssh keyed remote nodes), 
#. persist results into csv file (or maybe sqlite3 db). 
#. present these using highstocks or highchart from within sphinx 

   #. a cron controlled sphinx build donw on C2 ?

Thus can merge tens of daily tedious monitoring emails into a single
glance at a web page (which could be emailed as html), or even none once 
I trust the range checking.

   #. rendering charts to png non-trivial in absence of browser, so do checks on hub and email if notifications required


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


Rsync and scp Timeouts from C2R to N and N1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interactive scm-backup-rsync suffering timeouts whereas
not in the log from cron /var/scm/log/scm-backup-nightly.log 

#. seems to be network issue, cannot ping belle7 OR belle1 by name of number 
#. look for alternate backup targets... use mars


C2 timeout::


        [blyth@cms02 ~]$ ssh -v -v -v belle7.nuu.edu.tw
        OpenSSH_3.9p1, OpenSSL 0.9.7a Feb 19 2003
        debug1: Reading configuration data /home/blyth/.ssh/config
        debug1: Reading configuration data /etc/ssh/ssh_config
        debug1: Applying options for *
        debug2: ssh_connect: needpriv 0
        debug1: Connecting to belle7.nuu.edu.tw [203.64.184.126] port 22.
        ssh: connect to host 203.64.184.126 port 22: Connection timed out

C succeeds::


        [blyth@cms01 ~]$ ssh -v -v -v belle7.nuu.edu.tw
        OpenSSH_4.3p2-6.cern-hpn, OpenSSL 0.9.7a Feb 19 2003
        ssh(14212) debug1: Reading configuration data /home/blyth/.ssh/config
        ssh(14212) debug1: Reading configuration data /etc/ssh/ssh_config
        ssh(14212) debug1: Applying options for *
        ssh(14212) debug2: ssh_connect: needpriv 0
        ssh(14212) debug1: Connecting to belle7.nuu.edu.tw [203.64.184.126] port 22.
        ssh(14212) debug1: Connection established.
        ssh(14212) debug3: Not a RSA1 key file /home/blyth/.ssh/id_rsa.
        ...




compare openssh versions between C and C2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


::

        [blyth@cms02 ~]$ cat /etc/redhat-release 
        Scientific Linux SL release 4.5 (Beryllium)

        [blyth@cms02 ~]$ sudo yum list installed | grep ssh
        openssh.x86_64                           3.9p1-11.el4_7         installed       
        openssh-clients.x86_64                   3.9p1-11.el4_7         installed       
        openssh-server.x86_64                    3.9p1-11.el4_7         installed       


::

        [blyth@cms01 ~]$ cat /etc/redhat-release 
        Scientific Linux CERN SLC release 4.8 (Beryllium)

        [blyth@cms01 ~]$ sudo yum list installed | grep ssh
        Password:
        gsiopenssh.i386                          VDT1.6.0x86_rhas_4-1   installed       
        openssh.i386                             4.3p2-6.cern           installed       
        openssh-clients.i386                     4.3p2-6.cern           installed       
        openssh-server.i386                      4.3p2-6.cern           installed       



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




