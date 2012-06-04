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


Env
----


SVNlog not working as desired, has older revision stuck in craw ``02/08/12 12:12:49 (3 months ago)``

::

        [blyth@cms01 dybgaudi]$  svnlog --limit 1000000 -w 52 > 2012.txt 
        INFO:__main__:args [] opts {'author': None, 'loglevel': 'INFO', 'base': '.', 'limit': '1000000', 'weeks': '52', 'revision': None} 
        WARNING:__main__:reading from xmlcache /tmp/blyth/env/tools/svnlog/c1f4f22758749e1b672e413484c2144c.xmlcache 
        INFO:__main__:Info http://dayabay.ihep.ac.cn/svn/dybsvn /data/env/local/dyb/trunk/NuWa-trunk/dybgaudi 15835 
        INFO:__main__:SVNLog base http://dayabay.ihep.ac.cn/svn/dybsvn 
        WARNING:__main__:reading from xmlcache /tmp/blyth/env/tools/svnlog/cca5255cd16fa1ec9ac4421fb53e2950.xmlcache 
        WARNING:__main__:getElementsByTagName unexpected lec [] author 
        WARNING:__main__:getElementsByTagName unexpected lec [] author 


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


NUU Network
^^^^^^^^^^^^

::

        [blyth@belle7 env]$ svn up
        svn: OPTIONS of 'http://dayabay.phys.ntu.edu.tw/repos/env/trunk': could not connect to server (http://dayabay.phys.ntu.edu.tw)
        [blyth@belle7 env]$ 




Failed ssh from C2 to N  (other machines to N are OK)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. digraph:: foo


   subgraph nuu {
      style = "filled";
      N [label="belle7"] ;
      N1 [label="belle1"] ;
      label = "NUU" ; 
   };

   subgraph ntu {

      G [label="simon"];
      C [label="cms01"];
      C2 [label="cms02"];
      H1 [label="hep1"];
      H [label="hfag"];

      label = "NTU" ; 
   };


   G -> N;
   G -> N1;
   G -> C;
   G -> C2;
   G -> H;

   C -> C2;
   C -> H;
   C -> N;
   C -> N1 ;

   C2 -> N [label=hangs, color=red] ;
   C2 -> N1 [label=hangs, color=red] ;
   C2 -> C;
   C2 -> H1;
   C2 -> H [label=denied, color=purple] ;

   H -> N ;


Something is messed up with C2 and C2R ssh configuration.

::

        [blyth@hfag ssh]$  ssh -v -v 203.64.184.126
        OpenSSH_3.6.1p2, SSH protocols 1.5/2.0, OpenSSL 0x0090701f
        debug1: Reading configuration data /home/blyth/.ssh/config
        debug1: Reading configuration data /etc/ssh/ssh_config
        debug1: Applying options for *
        debug1: Rhosts Authentication disabled, originating port will not be trusted.
        debug2: ssh_connect: needpriv 0
        debug1: Connecting to 203.64.184.126 [203.64.184.126] port 22.
        debug1: Connection established.                     
        debug1: identity file /home/blyth/.ssh/identity type -1
        debug1: identity file /home/blyth/.ssh/id_rsa type -1


::


        [root@cms02 .ssh]# ssh -v -v -v N
        OpenSSH_3.9p1, OpenSSL 0.9.7a Feb 19 2003
        debug1: Reading configuration data /root/.ssh/config
        debug1: Applying options for N
        debug1: Reading configuration data /etc/ssh/ssh_config
        debug1: Applying options for *
        debug2: ssh_connect: needpriv 0
        debug1: Connecting to 203.64.184.126 [203.64.184.126] port 22.


Following the ``tail -f /var/log/secure`` on N while attempting to connect shows not getting through.
Nothing glaring in the iptables.







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




