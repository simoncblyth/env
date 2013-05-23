
MySQL repair table live
=========================


Extraction of dybdb1.ihep.ac.cn tarball onto belle7
-----------------------------------------------------

The tarball obtained by *coldcopy* on dybdb1 extracted onto belle7 without incident.

#. repeatable nature of the extraction means I can proceed with recovery efforts, without any need for caution

::

    [root@belle7 ~]# mysqlhotcopy.py -t 20130522_1541 --node dybdb1.ihep.ac.cn --rename tmp_ligs_offline_db_0 --containerdir /var/lib/mysql --ALLOWEXTRACT  tmp_ligs_offline_db examine extract
    2013-05-23 12:06:33,546 env.mysqlhotcopy.mysqlhotcopy INFO     /home/blyth/env/bin/mysqlhotcopy.py -t 20130522_1541 --node dybdb1.ihep.ac.cn --rename tmp_ligs_offline_db_0 --containerdir /var/lib/mysql --ALLOWEXTRACT tmp_ligs_offline_db examine extract
    2013-05-23 12:06:33,546 env.mysqlhotcopy.mysqlhotcopy INFO     backupdir /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db 
    2013-05-23 12:06:33,561 env.mysqlhotcopy.mysqlhotcopy INFO     failed to instanciate connection to database tmp_ligs_offline_db with exception Error 1049: Unknown database 'tmp_ligs_offline_db'  
    2013-05-23 12:06:33,561 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== examine 
    2013-05-23 12:06:33,562 env.mysqlhotcopy.tar INFO     examining /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130522_1541.tar.gz 
    2013-05-23 12:07:08,913 env.mysqlhotcopy.tar INFO     archive contains 7 items with commonprefix "" flattop True 
    2013-05-23 12:07:08,913 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_examine': 35.351444005966187, 'examine': 35.35143518447876} 
    2013-05-23 12:07:08,913 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== extract 
    2013-05-23 12:07:08,914 env.mysqlhotcopy.mysqlhotcopy WARNING  no valid db connection using static opts.mb_required 2000 
    2013-05-23 12:07:08,914 env.mysqlhotcopy.mysqlhotcopy INFO     sufficient free space,      required 2000 MB less than    free 494499.882812 MB 
    DO YOU REALLY WANT TO extract Tar /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130522_1541.tar.gz tmp_ligs_offline_db gz  into containerdir /var/lib/mysql    ? ENTER "YES" TO PROCEED : YES
    2013-05-23 12:07:48,589 env.mysqlhotcopy.mysqlhotcopy INFO     proceeding
    2013-05-23 12:07:48,589 env.mysqlhotcopy.mysqlhotcopy INFO     extract Tar /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130522_1541.tar.gz tmp_ligs_offline_db gz  into containerdir /var/lib/mysql   
    2013-05-23 12:07:48,589 env.mysqlhotcopy.tar INFO     _flat_extract opening tarfile /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130522_1541.tar.gz 
    2013-05-23 12:08:23,906 env.mysqlhotcopy.tar INFO     _flat_extract into target /var/lib/mysql/tmp_ligs_offline_db_0 for 7 members with toplevelname tmp_ligs_offline_db_0 
    2013-05-23 12:09:06,346 env.mysqlhotcopy.tar INFO     total 2429412
    -rw-rw---- 1 mysql mysql       8746 Feb  4 16:07 DqChannelStatus.frm
    -rw-rw---- 1 mysql mysql 1439608104 May 16 19:15 DqChannelStatus.MYD
    -rw-rw---- 1 mysql mysql 1024402432 May 16 19:42 DqChannelStatus.MYI
    -rw-rw---- 1 mysql mysql       8908 May 13 13:16 DqChannelStatusVld.frm
    -rw-rw---- 1 mysql mysql   17397375 May 20 06:26 DqChannelStatusVld.MYD
    -rw-rw---- 1 mysql mysql    3826688 May 20 06:26 DqChannelStatusVld.MYI

    2013-05-23 12:09:06,347 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_examine': 35.351444005966187, 'examine': 35.35143518447876, 'extract': 77.757769107818604, '_extract': 117.43390297889709} 
    [root@belle7 ~]# 



Repairing crashed DqChannelStatus table
------------------------------------------

#. crashed nature was propagated, as expected

::

    mysql> use tmp_ligs_offline_db_0 
    Reading table information for completion of table and column names
    You can turn off this feature to get a quicker startup with -A

    Database changed
    mysql> show tables ;
    +---------------------------------+
    | Tables_in_tmp_ligs_offline_db_0 |
    +---------------------------------+
    | DqChannelStatus                 | 
    | DqChannelStatusVld              | 
    +---------------------------------+
    2 rows in set (0.00 sec)

    mysql> select count(*) from DqChannelStatusVld   ;
    +----------+
    | count(*) |
    +----------+
    |   341125 | 
    +----------+
    1 row in set (0.00 sec)

    mysql> select count(*) from DqChannelStatus   ;
    ERROR 145 (HY000): Table './tmp_ligs_offline_db_0/DqChannelStatus' is marked as crashed and should be repaired
    mysql> 
    mysql> 


::

    mysql> check table  DqChannelStatus ;
    +---------------------------------------+-------+----------+-----------------------------------------------------------+
    | Table                                 | Op    | Msg_type | Msg_text                                                  |
    +---------------------------------------+-------+----------+-----------------------------------------------------------+
    | tmp_ligs_offline_db_0.DqChannelStatus | check | warning  | Table is marked as crashed                                | 
    | tmp_ligs_offline_db_0.DqChannelStatus | check | warning  | 3 clients are using or haven't closed the table properly  | 
    | tmp_ligs_offline_db_0.DqChannelStatus | check | error    | Record-count is not ok; is 65436731   Should be: 65436732 | 
    | tmp_ligs_offline_db_0.DqChannelStatus | check | warning  | Found 22 deleted space.   Should be 0                     | 
    | tmp_ligs_offline_db_0.DqChannelStatus | check | warning  | Found 1 deleted blocks       Should be: 0                 | 
    | tmp_ligs_offline_db_0.DqChannelStatus | check | error    | Corrupt                                                   | 
    +---------------------------------------+-------+----------+-----------------------------------------------------------+
    6 rows in set (25.21 sec)

::

    mysql> repair local table  DqChannelStatus ;
    +---------------------------------------+--------+----------+--------------------------------------------------+
    | Table                                 | Op     | Msg_type | Msg_text                                         |
    +---------------------------------------+--------+----------+--------------------------------------------------+
    | tmp_ligs_offline_db_0.DqChannelStatus | repair | warning  | Number of rows changed from 65436732 to 65436731 | 
    | tmp_ligs_offline_db_0.DqChannelStatus | repair | status   | OK                                               | 
    +---------------------------------------+--------+----------+--------------------------------------------------+
    2 rows in set (3 min 34.62 sec)




DqChannelStatus health checks
-------------------------------

::

    mysql> select count(*) from  DqChannelStatus ;
    +----------+
    | count(*) |
    +----------+
    | 65436731 | 
    +----------+
    1 row in set (0.06 sec)
        

::
 
    mysql> select max(SEQNO) from DqChannelStatus ;
    +------------+
    | max(SEQNO) |
    +------------+
    |     340817 | 
    +------------+
    1 row in set (0.00 sec)


    mysql> select min(SEQNO),max(SEQNO),min(ROW_COUNTER),max(ROW_COUNTER) from DqChannelStatus ;
    +------------+------------+------------------+------------------+
    | min(SEQNO) | max(SEQNO) | min(ROW_COUNTER) | max(ROW_COUNTER) |
    +------------+------------+------------------+------------------+
    |          1 |     340817 |                0 |              192 | 
    +------------+------------+------------------+------------------+
    1 row in set (26.50 sec)



::

    mysql> select ROW_COUNTER, count(*) as N from DqChannelStatus group by ROW_COUNTER ;
    +-------------+--------+
    | ROW_COUNTER | N      |
    +-------------+--------+
    |           0 |      1 | 
    |           1 | 340817 | 
    |           2 | 340817 | 
    |           3 | 340817 | 
    |           4 | 340817 | 
    ...
    |          52 | 340817 | 
    |          53 | 340817 | 
    |          54 | 340817 | 
    |          55 | 340817 | 
    |          56 | 340817 | 
    |          57 | 340817 | 
    |          58 | 340817 |      #  transition 
    |          59 | 340816 |      #  from single SEQNO partial payload 
    |          60 | 340816 | 
    |          61 | 340816 | 
    |          62 | 340816 | 
    |          63 | 340816 | 
    |          64 | 340816 | 
    |          65 | 340816 | 
    ...
    |         188 | 340816 | 
    |         189 | 340816 | 
    |         190 | 340816 | 
    |         191 | 340816 | 
    |         192 | 340816 | 
    +-------------+--------+
    193 rows in set (44.89 sec)


    mysql> /* excluding the bad SEQNO get back to regular structure */

    mysql>  select ROW_COUNTER, count(*) as N from DqChannelStatus where SEQNO != 323575 group by ROW_COUNTER ;
    +-------------+--------+
    | ROW_COUNTER | N      |
    +-------------+--------+
    |           1 | 340816 | 
    |           2 | 340816 | 
    |           3 | 340816 | 
    ...
    |         190 | 340816 | 
    |         191 | 340816 | 
    |         192 | 340816 | 
    +-------------+--------+
    192 rows in set (47.06 sec)

::

    mysql> select * from DqChannelStatus where ROW_COUNTER=0 ;                          
    +--------+-------------+-------+--------+-----------+--------+
    | SEQNO  | ROW_COUNTER | RUNNO | FILENO | CHANNELID | STATUS |
    +--------+-------------+-------+--------+-----------+--------+
    | 323575 |           0 |     0 |      0 |         0 |      0 | 
    +--------+-------------+-------+--------+-----------+--------+
    1 row in set (20.37 sec)

::

    mysql> select SEQNO, count(*) as N from DqChannelStatus group by SEQNO having N != 192 ; 
    +--------+----+
    | SEQNO  | N  |
    +--------+----+
    | 323575 | 59 | 
    +--------+----+
    1 row in set (25.72 sec)


    mysql> select * from  DqChannelStatus where SEQNO = 323575 ;                            
    +--------+-------------+-------+--------+-----------+--------+
    | SEQNO  | ROW_COUNTER | RUNNO | FILENO | CHANNELID | STATUS |
    +--------+-------------+-------+--------+-----------+--------+
    | 323575 |           0 |     0 |      0 |         0 |      0 | 
    | 323575 |           1 | 38347 |     43 |  33687041 |      1 | 
    | 323575 |           2 | 38347 |     43 |  33687042 |      1 | 
    | 323575 |           3 | 38347 |     43 |  33687043 |      1 | 
    | 323575 |           4 | 38347 |     43 |  33687044 |      1 | 
    | 323575 |           5 | 38347 |     43 |  33687045 |      1 | 
    | 323575 |           6 | 38347 |     43 |  33687046 |      1 | 
    ...
    | 323575 |          52 | 38347 |     43 |  33687812 |      1 | 
    | 323575 |          53 | 38347 |     43 |  33687813 |      1 | 
    | 323575 |          54 | 38347 |     43 |  33687814 |      1 | 
    | 323575 |          55 | 38347 |     43 |  33687815 |      1 | 
    | 323575 |          56 | 38347 |     43 |  33687816 |      1 | 
    | 323575 |          57 | 38347 |     43 |  33687817 |      1 | 
    | 323575 |          58 | 38347 |     43 |  33687818 |      1 | 
    +--------+-------------+-------+--------+-----------+--------+
    59 rows in set (0.00 sec)


Make mysqldump with bad SEQNO excluded
-----------------------------------------

* hmm, no locks are applied but the table is not active 

::

    [blyth@belle7 DybPython]$ dbdumpload.py tmp_ligs_offline_db_0 dump ~/tmp_ligs_offline_db_0.DqChannelStatus.sql --where 'SEQNO != 323575' --tables 'DqChannelStatus DqChannelStatusVld'         ## check the dump  command
    [blyth@belle7 DybPython]$ dbdumpload.py tmp_ligs_offline_db_0 dump ~/tmp_ligs_offline_db_0.DqChannelStatus.sql --where 'SEQNO != 323575' --tables 'DqChannelStatus DqChannelStatusVld' | sh    ## do it 

Huh mysqldump 2GB of SQL is quick::

    [blyth@belle7 DybPython]$ dbdumpload.py tmp_ligs_offline_db_0 dump ~/tmp_ligs_offline_db_0.DqChannelStatus.sql --where 'SEQNO != 323575' --tables 'DqChannelStatus DqChannelStatusVld' | sh 

    real    1m36.505s
    user    1m14.353s
    sys     0m6.705s
    [blyth@belle7 DybPython]$ 


Inspecting the dump file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 DybPython]$ du -h  ~/tmp_ligs_offline_db_0.DqChannelStatus.sql
    2.1G    /home/blyth/tmp_ligs_offline_db_0.DqChannelStatus.sql
    [blyth@belle7 DybPython]$ grep CREATE  ~/tmp_ligs_offline_db_0.DqChannelStatus.sql
    CREATE TABLE `DqChannelStatus` (
    CREATE TABLE `DqChannelStatusVld` (
    [blyth@belle7 DybPython]$ grep DROP  ~/tmp_ligs_offline_db_0.DqChannelStatus.sql
    [blyth@belle7 DybPython]$ 
    [blyth@belle7 DybPython]$ head -c 2000 ~/tmp_ligs_offline_db_0.DqChannelStatus.sql    ## looked OK,
    [blyth@belle7 DybPython]$ tail -c 2000 ~/tmp_ligs_offline_db_0.DqChannelStatus.sql    ## no truncation

                      
Recreate tables from the dump
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 ~]$ echo create database tmp_ligs_offline_db_1 | mysql 
    [blyth@belle7 ~]$ cat ~/tmp_ligs_offline_db_0.DqChannelStatus.sql |  mysql  tmp_ligs_offline_db_1     ## taking much longer to load than to dump, lunchtime



Full Server backup
--------------------

#. huh `ChannelQuality` continues to be updated

::

    mysql> show tables ;
    +-------------------------------+
    | Tables_in_tmp_ligs_offline_db |
    +-------------------------------+
    | ChannelQuality                | 
    | ChannelQualityVld             | 
    | DaqRawDataFileInfo            | 
    | DaqRawDataFileInfoVld         | 
    | DqChannel                     | 
    | DqChannelStatus               | 
    | DqChannelStatusVld            | 
    | DqChannelVld                  | 
    | LOCALSEQNO                    | 
    +-------------------------------+
    9 rows in set (0.07 sec)

    mysql> select * from DqChannelStatusVld order by SEQNO desc limit 1 ;
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | SEQNO  | TIMESTART           | TIMEEND             | SITEMASK | SIMMASK | SUBSITE | TASK | AGGREGATENO | VERSIONDATE         | INSERTDATE          |
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | 341125 | 2013-05-11 10:26:58 | 2013-05-11 10:43:11 |        4 |       1 |       1 |    0 |          -1 | 2013-05-11 10:26:58 | 2013-05-19 22:26:55 | 
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    1 row in set (0.06 sec)

    mysql> select * from DqChannelVld order by SEQNO desc limit 1 ;
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | SEQNO  | TIMESTART           | TIMEEND             | SITEMASK | SIMMASK | SUBSITE | TASK | AGGREGATENO | VERSIONDATE         | INSERTDATE          |
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | 341089 | 2013-05-11 10:26:58 | 2013-05-11 10:43:11 |        4 |       1 |       1 |    0 |          -1 | 2013-05-11 10:26:58 | 2013-05-19 22:26:54 | 
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    1 row in set (0.06 sec)

    mysql> select * from ChannelQualityVld order by SEQNO desc limit 1 ;
    +-------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | SEQNO | TIMESTART           | TIMEEND             | SITEMASK | SIMMASK | SUBSITE | TASK | AGGREGATENO | VERSIONDATE         | INSERTDATE          |
    +-------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    |  9093 | 2013-04-20 09:41:26 | 2038-01-19 03:14:07 |        4 |       1 |       4 |    0 |          -1 | 2012-12-07 07:13:46 | 2013-04-22 15:32:27 | 
    +-------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    1 row in set (0.07 sec)

    mysql> 


Before and during the table crash::


    mysql> select table_name,table_type, engine, round((data_length+index_length-data_free)/1024/1024,2) as MB  from information_schema.tables where table_schema = 'tmp_ligs_offline_db' ;
    +-----------------------+------------+-----------+---------+
    | table_name            | table_type | engine    | MB      |
    +-----------------------+------------+-----------+---------+
    | ChannelQuality        | BASE TABLE | MyISAM    |   47.31 | 
    | ChannelQualityVld     | BASE TABLE | MyISAM    |    0.53 | 
    | DaqRawDataFileInfo    | BASE TABLE | FEDERATED |   67.04 | 
    | DaqRawDataFileInfoVld | BASE TABLE | FEDERATED |   13.23 | 
    | DqChannel             | BASE TABLE | MyISAM    | 3570.58 | 
    | DqChannelStatus       | BASE TABLE | MyISAM    | 2338.56 | 
    | DqChannelStatusVld    | BASE TABLE | MyISAM    |   20.12 | 
    | DqChannelVld          | BASE TABLE | MyISAM    |   19.91 | 
    | LOCALSEQNO            | BASE TABLE | MyISAM    |    0.00 | 
    +-----------------------+------------+-----------+---------+
    9 rows in set (0.09 sec)

    mysql> select table_name,table_type, engine, round((data_length+index_length-data_free)/1024/1024,2) as MB  from information_schema.tables where table_schema = 'tmp_ligs_offline_db' ;
    +-----------------------+------------+-----------+---------+
    | table_name            | table_type | engine    | MB      |
    +-----------------------+------------+-----------+---------+
    | ChannelQuality        | BASE TABLE | MyISAM    |   47.31 | 
    | ChannelQualityVld     | BASE TABLE | MyISAM    |    0.53 | 
    | DaqRawDataFileInfo    | BASE TABLE | FEDERATED |   67.73 | 
    | DaqRawDataFileInfoVld | BASE TABLE | FEDERATED |   13.37 | 
    | DqChannel             | BASE TABLE | MyISAM    | 3591.27 | 
    | DqChannelStatus       | BASE TABLE | NULL      |    NULL | 
    | DqChannelStatusVld    | BASE TABLE | MyISAM    |   20.24 | 
    | DqChannelVld          | BASE TABLE | MyISAM    |   20.03 | 
    | LOCALSEQNO            | BASE TABLE | MyISAM    |    0.00 | 
    +-----------------------+------------+-----------+---------+
    9 rows in set (0.08 sec)








Repair on primary server, or propagate the fixed ?
----------------------------------------------------

