cms01
=======

Jul 30, 2015 Manual Stop prior to powercut
-------------------------------------------

Old HFAG server was still running (unattended)::

    [blyth@cms01 ~]$ sudo /usr/sbin/apachectl stop

    [blyth@cms01 ~]$ sudo /sbin/shutdown -h now "goin down"

    Broadcast message from root (pts/0) (Thu Jul 30 15:09:43 2015):

    goin down 
    The system is going down for system halt NOW!
    [blyth@cms01 ~]$ 






Dec 26, 2014
--------------

Low space again.

::

    delta:~ blyth$ scp C:.env/envmon.sqlite .env/C_envmon.sqlite

::

    monitordb.sh gb_free --limit 72 --level debug   
      ## daily tidal range of ~13 GB, but longterm slow trend down



Oct 23, 2014
-------------

Smth changed, getting low on space::

    [blyth@cms01 /]$ sudo du -hs * 
    Password:
    113G    data
    72G home
    8.9G    var
    2.3G    usr
    ..
    [blyth@cms01 /]$ date
    Thu Oct 23 11:57:20 CST 2014


Looks to be dybsvn::

    [blyth@cms01 alt.backup]$ find . -name '*.gz' -exec du -h {} \;
    4.0G    ./dayabay/svn/dybaux/2014/10/22/091002/dybaux-5486.tar.gz
    4.0G    ./dayabay/svn/dybaux/2014/05/10/091002/dybaux-5484.tar.gz
    1.4G    ./dayabay/svn/dybaux/2014/05/11/091002/dybaux-5484.tar.gz
    7.1G    ./dayabay/svn/dybsvn/2014/10/21/091002/dybsvn-23456.tar.gz
    7.1G    ./dayabay/svn/dybsvn/2014/10/22/091002/dybsvn-23458.tar.gz
    7.1G    ./dayabay/svn/dybsvn/2014/10/20/091002/dybsvn-23443.tar.gz
    47M ./dayabay/svn/toysvn/2014/10/22/091002/toysvn-185.tar.gz
    47M ./dayabay/svn/toysvn/2014/05/09/091002/toysvn-185.tar.gz
    47M ./dayabay/svn/toysvn/2014/05/10/091002/toysvn-185.tar.gz
    3.0M    ./dayabay/tracs/dybaux/2014/10/22/091002/dybaux.tar.gz
    2.8M    ./dayabay/tracs/dybaux/2014/05/10/091002/dybaux.tar.gz
    2.8M    ./dayabay/tracs/dybaux/2014/05/11/091002/dybaux.tar.gz
    1.7G    ./dayabay/tracs/dybsvn/2014/10/21/091002/dybsvn.tar.gz
    1.7G    ./dayabay/tracs/dybsvn/2014/10/22/091002/dybsvn.tar.gz
    1.7G    ./dayabay/tracs/dybsvn/2014/10/20/091002/dybsvn.tar.gz
    1.0M    ./dayabay/tracs/toysvn/2014/10/22/091002/toysvn.tar.gz
    884K    ./dayabay/tracs/toysvn/2014/05/10/091002/toysvn.tar.gz
    884K    ./dayabay/tracs/toysvn/2014/05/11/091002/toysvn.tar.gz
    12K ./dayabay/folders/svnsetup/2014/10/21/091002/svnsetup.tar.gz
    12K ./dayabay/folders/svnsetup/2014/10/22/091002/svnsetup.tar.gz
    12K ./dayabay/folders/svnsetup/2014/10/20/091002/svnsetup.tar.gz
    24K ./dayabay/repos/newtest/2014/10/22/091002/newtest-0.tar.gz
    24K ./dayabay/repos/newtest/2014/05/10/091002/newtest-0.tar.gz
    24K ./dayabay/repos/newtest/2014/05/11/091002/newtest-0.tar.gz




cms01 /data mount
--------------------

Following reboots cms01 /data mount must be done manually.::

    [blyth@cms01 ~]$ df -h
    Filesystem            Size  Used Avail Use% Mounted on
    /dev/hda2              20G   17G  1.4G  93% /
    /dev/hda1             251M   28M  210M  12% /boot
    none                 1013M     0 1013M   0% /dev/shm
    /dev/hda5              77G   71G  2.5G  97% /home
    [blyth@cms01 ~]$ 
    [blyth@cms01 ~]$ 
    [blyth@cms01 ~]$ sudo mount /data
    Password:
    [blyth@cms01 ~]$ df -h
    Filesystem            Size  Used Avail Use% Mounted on
    /dev/hda2              20G   17G  1.4G  93% /
    /dev/hda1             251M   28M  210M  12% /boot
    none                 1013M     0 1013M   0% /dev/shm
    /dev/hda5              77G   71G  2.5G  97% /home
    /dev/hda6             132G  106G   20G  85% /data
    [blyth@cms01 ~]$ 


jun 29 2014, unusual disk usage bump : EXPLAINED
---------------------------------------------------


::


    cms01.phys.ntu.edu.tw : valmon.py -s diskmon : 2014-06-29T18:52:01 :    gb_free > 10 : False 

      gb_free > 10                             : False 

      summary                                  : all:1 True:0 False:1  
      context                                  : {'gb_free': 8.4199999999999999} 


    % ~/.env.cnf blyth@cms01.phys.ntu.edu.tw 
    [diskmon]
    return = dict 
    dbpath = ~/.env/envmon.sqlite 
    cmd = disk_usage.py /data 
    tn = diskmon 
    valmon_version = 0.2 
    email = blyth@hep1.phys.ntu.edu.tw 
    constraints = ( gb_free > 10, ) 

    date                 runtime            ret                                                                                         val         rc        
    -------------------  -----------------  ------------------------------------------------------------------------------------------  ---------- ----------
    2014-06-29T18:52:01  0.144702196121216  {'gb_total': '131.74', 'gb_free': '8.42', 'percent_free': '6.39', 'percent_used': '88.53'}  0.0         0         
    2014-06-29T17:52:02  0.044398069381713  {'gb_total': '131.74', 'gb_free': '14.44', 'percent_free': '10.96', 'percent_used': '83.96  0.0         0         
    2014-06-29T16:52:01  0.044739007949829  {'gb_total': '131.74', 'gb_free': '14.44', 'percent_free': '10.96', 'percent_used': '83.96  0.0         0         
    2014-06-29T15:52:01  0.215399980545044  {'gb_total': '131.74', 'gb_free': '14.44', 'percent_free': '10.96', 'percent_used': '83.96  0.0         0         
    2014-06-29T14:52:01  0.046348810195922  {'gb_total': '131.74', 'gb_free': '14.75', 'percent_free': '11.19', 'percent_used': '83.73  0.0         0         




::

    cms01.phys.ntu.edu.tw : valmon.py -s diskmon : 2014-06-28T18:52:01 :    gb_free > 10 : False 

      gb_free > 10                             : False 

      summary                                  : all:1 True:0 False:1  
      context                                  : {'gb_free': 9.8100000000000005} 


    % ~/.env.cnf blyth@cms01.phys.ntu.edu.tw 
    [diskmon]
    return = dict 
    dbpath = ~/.env/envmon.sqlite 
    cmd = disk_usage.py /data 
    tn = diskmon 
    valmon_version = 0.2 
    email = blyth@hep1.phys.ntu.edu.tw 
    constraints = ( gb_free > 10, ) 


    date                 runtime            ret                                                                                         val         rc        
    -------------------  -----------------  ------------------------------------------------------------------------------------------  ---------- ----------
    2014-06-28T18:52:01  0.134742021560669  {'gb_total': '131.74', 'gb_free': '9.81', 'percent_free': '7.45', 'percent_used': '87.47'}  0.0         0         
    2014-06-28T17:52:01  0.045220851898193  {'gb_total': '131.74', 'gb_free': '15.84', 'percent_free': '12.02', 'percent_used': '82.90  0.0         0         
    2014-06-28T16:52:01  0.044707059860229  {'gb_total': '131.74', 'gb_free': '15.84', 'percent_free': '12.02', 'percent_used': '82.90  0.0         0         
    2014-06-28T15:52:01  0.265349864959717  {'gb_total': '131.74', 'gb_free': '15.84', 'percent_free': '12.02', 'percent_used': '82.90  0.0         0         





valmon disk space monitoring indicates unexplained 5GB bump 
Saturday evening between 2014-06-28T17:52:01  and 2014-06-28T18:52:01 

Find and kill 2 stuck curl processes (env apache monitoring).



No jump in offline_db tarball sizes::

    [blyth@cms01 dbbackup]$ find . -name 'offline_db.sql.gz' -exec du -h {} \;
    145M    ./rsync/dybdb1.ihep.ac.cn/20131025/offline_db.sql.gz
    145M    ./rsync/dybdb1.ihep.ac.cn/20131026/offline_db.sql.gz
    145M    ./rsync/dybdb1.ihep.ac.cn/20131028/offline_db.sql.gz
    145M    ./rsync/dybdb1.ihep.ac.cn/20131029/offline_db.sql.gz
    145M    ./rsync/dybdb1.ihep.ac.cn/20131027/offline_db.sql.gz
    145M    ./rsync/dybdb1.ihep.ac.cn/20131031/offline_db.sql.gz
    145M    ./rsync/dybdb1.ihep.ac.cn/20131030/offline_db.sql.gz
    168M    ./rsync/dybdb2.ihep.ac.cn/20140629/offline_db.sql.gz
    168M    ./rsync/dybdb2.ihep.ac.cn/20140627/offline_db.sql.gz
    168M    ./rsync/dybdb2.ihep.ac.cn/20140628/offline_db.sql.gz
    168M    ./rsync/dybdb2.ihep.ac.cn/20140626/offline_db.sql.gz
    168M    ./rsync/dybdb2.ihep.ac.cn/20140630/offline_db.sql.gz
    168M    ./rsync/dybdb2.ihep.ac.cn/20140624/offline_db.sql.gz
    168M    ./rsync/dybdb2.ihep.ac.cn/20140625/offline_db.sql.gz



Found it, alt.backup for dybsvn has started working again writing 6.3G per day::

    [blyth@cms01 var]$ find . -name '*.gz' -exec du -h {} \;
    4.0G    ./scm/alt.backup/dayabay/svn/dybaux/2014/05/09/091002/dybaux-5484.tar.gz
    4.0G    ./scm/alt.backup/dayabay/svn/dybaux/2014/05/10/091002/dybaux-5484.tar.gz
    1.4G    ./scm/alt.backup/dayabay/svn/dybaux/2014/05/11/091002/dybaux-5484.tar.gz
    6.3G    ./scm/alt.backup/dayabay/svn/dybsvn/2014/06/27/091002/dybsvn-23074.tar.gz
    6.3G    ./scm/alt.backup/dayabay/svn/dybsvn/2014/06/29/091001/dybsvn-23074.tar.gz
    6.3G    ./scm/alt.backup/dayabay/svn/dybsvn/2014/06/28/091002/dybsvn-23074.tar.gz
    47M ./scm/alt.backup/dayabay/svn/toysvn/2014/05/09/091002/toysvn-185.tar.gz
    47M ./scm/alt.backup/dayabay/svn/toysvn/2014/05/10/091002/toysvn-185.tar.gz
    47M ./scm/alt.backup/dayabay/svn/toysvn/2014/05/08/091002/toysvn-185.tar.gz
    2.8M    ./scm/alt.backup/dayabay/tracs/dybaux/2014/05/09/091002/dybaux.tar.gz
    2.8M    ./scm/alt.backup/dayabay/tracs/dybaux/2014/05/10/091002/dybaux.tar.gz
    2.8M    ./scm/alt.backup/dayabay/tracs/dybaux/2014/05/11/091002/dybaux.tar.gz
    1.6G    ./scm/alt.backup/dayabay/tracs/dybsvn/2014/06/27/091002/dybsvn.tar.gz


Will need some cleanup of old tarballs to support this.


