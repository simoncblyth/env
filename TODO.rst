TODO
=====

env scm-backup
------------------

#. move to full nodename for /var/scm/backup/NODENAME  (otherwise '''dayabay''' clashes for Jimmy)
#. make tarball DNA mismatches get reported more loudly (maybe adopt into the fabric way of doing things) 

   * :doc:`scm/dev`

slow trac due to backups
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. record backup time and rsync time, and make those graphically accessible : suspect slow Trac performance at ~13:30 is due to backups hogging CPU 

::

    [dayabay] /home/blyth > ps aux | grep scm ; date
    root     11461  0.0  0.0  3280  848 ?        Ss   12:00   0:00 /bin/bash -c /home/scm/etc/scm_backup.cron > /home/scm/log/scm-backup-nightly-$(date +"%d").log 2>&1
    root     11463  0.0  0.0  5288 3148 ?        S    12:00   0:00 /bin/bash /home/scm/etc/scm_backup.cron
    root     15386 22.2  0.0  3908 1708 ?        S    12:39  11:00 rsync -e ssh --delete-after --stats -razvt /home/scm/backup/dayabay SDU:/raid4/dybsdu/dybbackup/var/scm/backup/
    root     15387  6.2  0.0  7264 4688 ?        S    12:39   3:07 ssh SDU rsync --server -vlogDtprz --delete --delete-after . /raid4/dybsdu/dybbackup/var/scm/backup/
    blyth    16179  0.0  0.0  4692  584 pts/1    S+   13:28   0:00 grep scm
    Fri Nov 30 13:28:52 CST 2012    

    [dayabay] /home/blyth > ps aux | grep scm ; date
    blyth    16375  0.0  0.0  4752  572 pts/1    S+   13:31   0:00 grep scm
    Fri Nov 30 13:31:42 CST 2012

The backup looks to have taked 40 min and rsync 50min  


Repository Migration to shared services ?
------------------------------------------

 * investigate how difficult to migrate Trac+SVN+TracWiki into git+github+Sphinx
    
     * especially needed for NTU hosted **env**, **tracdev**, **heprez** as need long term home that will survive me
     
     * aberdeen repository is fat : and cannot be open source ?  


Alternatives to Git + Github
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * http://en.wikipedia.org/wiki/Comparison_of_open_source_software_hosting_facilities
 * https://bitbucket.org/  offers unlimited git or hg public and private repos, free for up to 5 users




