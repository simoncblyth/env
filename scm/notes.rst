Notes
======

tgz.py 
        provides `TGZ` :  interface between file system tgz directories and SQLite representation of that 





C2 backups sent to C
------------------------

Cron job runs on C2 at 16:10 TW time::

    [blyth@cms02 env]$ ll $LOCAL_BASE/env/scm/scm_backup_monitor.db
    -rw-r--r--  1 root root 1136640 Nov 13 16:17 /data/env/local/env/scm/scm_backup_monitor.db
    [blyth@cms02 env]$ 
    [blyth@cms02 env]$ date 
    Tue Nov 13 16:38:00 CST 2012

Starting point is the file system strings::

    [blyth@cms02 cms02]$ ll tracs/env/
    total 52
    drwxr-xr-x  10 blyth blyth 4096 Dec  1  2009 2009
    drwxr-xr-x  14 blyth blyth 4096 Dec  1  2010 2010
    drwxr-xr-x  14 blyth blyth 4096 Dec  1  2011 2011
    drwxr-xr-x   8 blyth blyth 4096 May  8  2012 ..
    drwxr-xr-x  13 blyth blyth 4096 Nov  1 16:12 2012
    lrwxrwxrwx   1 root  root    17 Nov 13 16:13 last -> 2012/11/13/161023
    drwxr-xr-x   6 blyth blyth 4096 Nov 13 16:13 .

    [blyth@cms02 cms02]$ find . -name '*.tar.gz' | grep 2012/11/13
    ./folders/svnsetup/2012/11/13/161023/svnsetup.tar.gz
    ./tracs/aberdeen/2012/11/13/161023/aberdeen.tar.gz
    ./tracs/tracdev/2012/11/13/161023/tracdev.tar.gz
    ./tracs/newtest/2012/11/13/161023/newtest.tar.gz
    ./tracs/env/2012/11/13/161023/env.tar.gz
    ./tracs/heprez/2012/11/13/161023/heprez.tar.gz
    ./repos/aberdeen/2012/11/13/161023/aberdeen-1789.tar.gz
    ./repos/tracdev/2012/11/13/161023/tracdev-125.tar.gz
    ./repos/newtest/2012/11/13/161023/newtest-20.tar.gz
    ./repos/data/2012/11/13/161023/data-23.tar.gz
    ./repos/env/2012/11/13/161023/env-3591.tar.gz
    ./repos/heprez/2012/11/13/161023/heprez-822.tar.gz


Which are parsed by `datepath.py` into datetime and written into SQLite DB as strings::


    g4pb-3:scm blyth$ sqlite3 $LOCAL_BASE/env/scm/scm_backup_monitor.db

    sqlite> select date, strftime("%s",date)*1000  from tgzs order by date desc limit 10 ;
    date                 strftime("%s",date)*1000
    -------------------  ------------------------
    2012-07-13T10:20:03  1342174803000           
    2012-07-13T10:20:03  1342174803000           
    2012-07-13T10:20:03  1342174803000           
    2012-07-13T10:20:03  1342174803000           
    2012-07-13T10:20:03  1342174803000           
    2012-07-13T10:20:03  1342174803000           
    2012-07-13T10:20:03  1342174803000           
    2012-07-12T14:10:03  1342102203000           
    2012-07-12T14:10:03  1342102203000           
    2012-07-12T14:10:03  1342102203000           


In the DB::

    [blyth@cms02 env]$ sqlite3 $LOCAL_BASE/env/scm/scm_backup_monitor.db
    SQLite version 3.3.6
    Enter ".help" for instructions
    sqlite> select count(*) from tgzs ;
    3616
    sqlite> select min(date), max(date) from tgzs ;
    2012-04-28T13:01:07|2012-11-13T16:10:23
    sqlite> 

    [blyth@cms02 cms02]$  sqlite3 $LOCAL_BASE/env/scm/scm_backup_monitor.db
    SQLite version 3.3.6
    Enter ".help" for instructions
    sqlite> select * from tgzs order by date desc limit 100 ;
    2012-11-13T16:10:23|C|2.0|/data/var/scm/backup/cms02/tracs/aberdeen|C:/data/var/scm/backup/cms02/tracs/aberdeen/2012/11/13/161023/aberdeen.tar.gz
    2012-11-13T16:10:23|C|9.0|/data/var/scm/backup/cms02/tracs/heprez|C:/data/var/scm/backup/cms02/tracs/heprez/2012/11/13/161023/heprez.tar.gz
    2012-11-13T16:10:23|C|1.0|/data/var/scm/backup/cms02/tracs/newtest|C:/data/var/scm/backup/cms02/tracs/newtest/2012/11/13/161023/newtest.tar.gz
    2012-11-13T16:10:23|C|52.0|/data/var/scm/backup/cms02/tracs/env|C:/data/var/scm/backup/cms02/tracs/env/2012/11/13/161023/env.tar.gz
    2012-11-13T16:10:23|C|1.0|/data/var/scm/backup/cms02/tracs/tracdev|C:/data/var/scm/backup/cms02/tracs/tracdev/2012/11/13/161023/tracdev.tar.gz
    2012-11-13T16:10:23|C|1.0|/data/var/scm/backup/cms02/folders/svnsetup|C:/data/var/scm/backup/cms02/folders/svnsetup/2012/11/13/161023/svnsetup.tar.gz
    2012-11-13T16:10:23|C|184.0|/data/var/scm/backup/cms02/repos/aberdeen|C:/data/var/scm/backup/cms02/repos/aberdeen/2012/11/13/161023/aberdeen-1789.tar.gz
    2012-11-13T16:10:23|C|1.0|/data/var/scm/backup/cms02/repos/data|C:/data/var/scm/backup/cms02/repos/data/2012/11/13/161023/data-23.tar.gz
    2012-11-13T16:10:23|C|5.0|/data/var/scm/backup/cms02/repos/heprez|C:/data/var/scm/backup/cms02/repos/heprez/2012/11/13/161023/heprez-822.tar.gz
    2012-11-13T16:10:23|C|4.0|/data/var/scm/backup/cms02/repos/newtest|C:/data/var/scm/backup/cms02/repos/newtest/2012/11/13/161023/newtest-20.tar.gz
    2012-11-13T16:10:23|C|10.0|/data/var/scm/backup/cms02/repos/env|C:/data/var/scm/backup/cms02/repos/env/2012/11/13/161023/env-3591.tar.gz
    2012-11-13T16:10:23|C|1.0|/data/var/scm/backup/cms02/repos/tracdev|C:/data/var/scm/backup/cms02/repos/tracdev/2012/11/13/161023/tracdev-125.tar.gz
    2012-11-13T16:10:23|H1|1.0|/home/hep/blyth/var/scm/backup/cms02/tracs/tracdev|H1:/home/hep/blyth/var/scm/backup/cms02/tracs/tracdev/2012/11/13/161023/tracdev.tar.gz
    2012-11-13T16:10:23|H1|2.0|/home/hep/blyth/var/scm/backup/cms02/tracs/aberdeen|H1:/home/hep/blyth/var/scm/backup/cms02/tracs/aberdeen/2012/11/13/161023/aberdeen.tar.gz
    2012-11-13T16:10:23|H1|52.0|/home/hep/blyth/var/scm/backup/cms02/tracs/env|H1:/home/hep/blyth/var/scm/backup/cms02/tracs/env/2012/11/13/161023/env.tar.gz
    2012-11-13T16:10:23|H1|9.0|/home/hep/blyth/var/scm/backup/cms02/tracs/heprez|H1:/home/hep/blyth/var/scm/backup/cms02/tracs/heprez/2012/11/13/161023/heprez.tar.gz
    2012-11-13T16:10:23|H1|1.0|/home/hep/blyth/var/scm/backup/cms02/tracs/newtest|H1:/home/hep/blyth/var/scm/backup/cms02/tracs/newtest/2012/11/13/161023/newtest.tar.gz
    2012-11-13T16:10:23|H1|1.0|/home/hep/blyth/var/scm/backup/cms02/folders/svnsetup|H1:/home/hep/blyth/var/scm/backup/cms02/folders/svnsetup/2012/11/13/161023/svnsetup.tar.gz
    2012-11-13T16:10:23|H1|1.0|/home/hep/blyth/var/scm/backup/cms02/repos/tracdev|H1:/home/hep/blyth/var/scm/backup/cms02/repos/tracdev/2012/11/13/161023/tracdev-125.tar.gz
    2012-11-13T16:10:23|H1|184.0|/home/hep/blyth/var/scm/backup/cms02/repos/aberdeen|H1:/home/hep/blyth/var/scm/backup/cms02/repos/aberdeen/2012/11/13/161023/aberdeen-1789.tar.gz
    2012-11-13T16:10:23|H1|1.0|/home/hep/blyth/var/scm/backup/cms02/repos/data|H1:/home/hep/blyth/var/scm/backup/cms02/repos/data/2012/11/13/161023/data-23.tar.gz
    2012-11-13T16:10:23|H1|10.0|/home/hep/blyth/var/scm/backup/cms02/repos/env|H1:/home/hep/blyth/var/scm/backup/cms02/repos/env/2012/11/13/161023/env-3591.tar.gz
    2012-11-13T16:10:23|H1|5.0|/home/hep/blyth/var/scm/backup/cms02/repos/heprez|H1:/home/hep/blyth/var/scm/backup/cms02/repos/heprez/2012/11/13/161023/heprez-822.tar.gz
    2012-11-13T16:10:23|H1|4.0|/home/hep/blyth/var/scm/backup/cms02/repos/newtest|H1:/home/hep/blyth/var/scm/backup/cms02/repos/newtest/2012/11/13/161023/newtest-20.tar.gz
    2012-11-12T16:10:25|C|2.0|/data/var/scm/backup/cms02/tracs/aberdeen|C:/data/var/scm/backup/cms02/tracs/aberdeen/2012/11/12/161025/aberdeen.tar.gz
    2012-11-12T16:10:25|C|9.0|/data/var/scm/backup/cms02/tracs/heprez|C:/data/var/scm/backup/cms02/tracs/heprez/2012/11/12/161025/heprez.tar.gz


SQLite not pulling any tricks::

    [blyth@cms02 cms02]$ echo "select strftime('%s','1970-01-01T00:00:00');" | sqlite3
    0


All so far in localtime

   http://dayabay.phys.ntu.edu.tw/data/scm_backup_monitor_C.json

Now in the JSON, something is shunting it forwards 8hrs (local interpreted as UTC)::

    imon:scm blyth$ ./tgzmon.py 
    INFO:__main__:series OK                   last [1352764800000L, 12] ts 1352764800 stamp 2012-11-13 08:00:00 
    INFO:__main__:series tracs/aberdeen       last [1352823023000L, 20.0] ts 1352823023 stamp 2012-11-14 00:10:23 
    INFO:__main__:series tracs/heprez         last [1352823023000L, 90.0] ts 1352823023 stamp 2012-11-14 00:10:23 
    INFO:__main__:series tracs/env            last [1352823023000L, 52.0] ts 1352823023 stamp 2012-11-14 00:10:23 
    INFO:__main__:series tracs/tracdev        last [1352823023000L, 10.0] ts 1352823023 stamp 2012-11-14 00:10:23 
    INFO:__main__:series repos/aberdeen       last [1352823023000L, 184.0] ts 1352823023 stamp 2012-11-14 00:10:23 
    INFO:__main__:series repos/heprez         last [1352823023000L, 50.0] ts 1352823023 stamp 2012-11-14 00:10:23 
    INFO:__main__:series repos/env            last [1352823023000L, 100.0] ts 1352823023 stamp 2012-11-14 00:10:23 
    INFO:__main__:series repos/tracdev        last [1352823023000L, 10.0] ts 1352823023 stamp 2012-11-14 00:10:23 
    INFO:env.plot.highmon:no violations, not sending email
    
datetime.fromtimestamp returns local
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These ms timestamps get converted back to datetimes by::

    In [82]: datetime.fromtimestamp(1342102203000/1000)
    Out[82]: datetime.datetime(2012, 7, 12, 22, 10, 3)

    In [86]: datetime.fromtimestamp(0)               ## treats stamp as local
    Out[86]: datetime.datetime(1970, 1, 1, 8, 0)

    In [176]: datetime.utcfromtimestamp(0)           ## treats stamp as UTC 
    Out[176]: datetime.datetime(1970, 1, 1, 0, 0)



sqlite not offsetting
~~~~~~~~~~~~~~~~~~~~~~~~

Not from SQLite::

    sqlite> select strftime("%s","1970-01-01T00:00:00") ;
    strftime("%s","1970-01-01T00:00:00")
    ------------------------------------
    0                                   
    sqlite> select strftime("%s","1970-01-01T08:00:00") ;
    strftime("%s","1970-01-01T08:00:00")
    ------------------------------------
    28800                       



