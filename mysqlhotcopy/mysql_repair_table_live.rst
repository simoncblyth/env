
MySQL repair table live
=========================

.. contents:: :local:

History Timeline has Moved 
----------------------------

* http://belle7.nuu.edu.tw/oum/aop/dbrepair/history/
* http://dayabay.bnl.gov/oum/aop/dbrepair/history/


hotcopy crash
~~~~~~~~~~~~~~~~
::

    2013-05-20 11:15:01,333 __main__ INFO     proceed with MySQLHotCopy /usr/bin/mysqlhotcopy  tmp_ligs_offline_db /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130520_1115   
    340     DBD::mysql::db do failed: Table './tmp_ligs_offline_db/DqChannelStatus' is marked as crashed and should be repaired at /usr/bin/mysqlhotcopy line 467.   


all queries fail for crashed table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    mysql> select count(*) from DqChannelStatus   ;
    ERROR 145 (HY000): Table './tmp_ligs_offline_db_0/DqChannelStatus' is marked as crashed and should be repaired
 
relevant INSERTs
~~~~~~~~~~~~~~~~~

::

    mysql> select * from  tmp_ligs_offline_db_0.DqChannelStatusVld where SEQNO in (323575,340817,341125) ;
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | SEQNO  | TIMESTART           | TIMEEND             | SITEMASK | SIMMASK | SUBSITE | TASK | AGGREGATENO | VERSIONDATE         | INSERTDATE          |
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | 323575 | 2013-04-01 09:59:43 | 2013-04-01 10:12:13 |        2 |       1 |       2 |    0 |          -1 | 2013-04-01 09:59:43 | 2013-04-30 10:14:06 |   ## corrupted SEQNO
    | 340817 | 2013-05-16 08:11:38 | 2013-05-16 08:24:05 |        2 |       1 |       1 |    0 |          -1 | 2013-05-16 08:11:38 | 2013-05-16 11:14:59 |   ## max SEQNO in payload table DqChannelStatus
    | 341125 | 2013-05-11 10:26:58 | 2013-05-11 10:43:11 |        4 |       1 |       1 |    0 |          -1 | 2013-05-11 10:26:58 | 2013-05-19 22:26:55 |   ## max SEQNO in validity table DqChannelStatus
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    3 rows in set (0.00 sec)
 


Extraction of dybdb1.ihep.ac.cn tarball onto belle7 into `tmp_ligs_offline_db_0`
-----------------------------------------------------------------------------------

The tarball obtained by *coldcopy* on dybdb1 extracted onto belle7 without incident. The command 
creates the DB `tmp_ligs_offline_db_0`

* repeatable nature of the extraction means I can proceed with recovery efforts, without any need for caution

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


Repairing crashed DqChannelStatus table in `tmp_ligs_offline_db_0` 
--------------------------------------------------------------------

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



Using local prevents replication, if were in a replication chain:: 

    mysql> repair local table  DqChannelStatus ;
    +---------------------------------------+--------+----------+--------------------------------------------------+
    | Table                                 | Op     | Msg_type | Msg_text                                         |
    +---------------------------------------+--------+----------+--------------------------------------------------+
    | tmp_ligs_offline_db_0.DqChannelStatus | repair | warning  | Number of rows changed from 65436732 to 65436731 | 
    | tmp_ligs_offline_db_0.DqChannelStatus | repair | status   | OK                                               | 
    +---------------------------------------+--------+----------+--------------------------------------------------+
    2 rows in set (3 min 34.62 sec)

Wouldnt skipping things from replication cause divergence ? Good thing this table is excluded from replication.


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

Huh mysqldump 2GB of SQL is very quick::

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

    
    
Digest, compress, publish, test url and digest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 ~]$ md5sum tmp_ligs_offline_db_0.DqChannelStatus.sql
    46b747d88ad74caa4b1d21be600265a4  tmp_ligs_offline_db_0.DqChannelStatus.sql
    [blyth@belle7 ~]$ gzip -c tmp_ligs_offline_db_0.DqChannelStatus.sql > tmp_ligs_offline_db_0.DqChannelStatus.sql.gz
    [blyth@belle7 ~]$ du -hs tmp_ligs_offline_db_0.DqChannelStatus.sql*
    2.1G    tmp_ligs_offline_db_0.DqChannelStatus.sql
    335M    tmp_ligs_offline_db_0.DqChannelStatus.sql.gz
    [blyth@belle7 ~]$ sudo mv tmp_ligs_offline_db_0.DqChannelStatus.sql.gz $(nginx-htdocs)/data/
    [blyth@belle7 ~]$ cd /tmp
    [blyth@belle7 tmp]$ curl -O http://belle7.nuu.edu.tw/data/tmp_ligs_offline_db_0.DqChannelStatus.sql.gz
    [blyth@belle7 tmp]$ du -h tmp_ligs_offline_db_0.DqChannelStatus.sql.gz
    335M    tmp_ligs_offline_db_0.DqChannelStatus.sql.gz
    [blyth@belle7 tmp]$ gunzip tmp_ligs_offline_db_0.DqChannelStatus.sql.gz
    [blyth@belle7 tmp]$ md5sum tmp_ligs_offline_db_0.DqChannelStatus.sql
    46b747d88ad74caa4b1d21be600265a4  tmp_ligs_offline_db_0.DqChannelStatus.sql

                 
Features of the dump `tmp_ligs_offline_db_0.DqChannelStatus.sql`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. bad SEQNO 323575 is excluded
#. 308 SEQNO `> 340817` are validity only, namely `340818:341125` 

                  
Recreate tables from the dump into `tmp_ligs_offline_db_1`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 ~]$ echo create database tmp_ligs_offline_db_1 | mysql 
    [blyth@belle7 ~]$ cat ~/tmp_ligs_offline_db_0.DqChannelStatus.sql |  mysql  tmp_ligs_offline_db_1     ## taking much longer to load than to dump, lunchtime


* looks like Vld continues to be written after the payload crashed ??

::

    mysql> show tables ;
    +---------------------------------+
    | Tables_in_tmp_ligs_offline_db_1 |
    +---------------------------------+
    | DqChannelStatus                 | 
    | DqChannelStatusVld              | 
    +---------------------------------+
    2 rows in set (0.00 sec)

    mysql> select count(*) from DqChannelStatus  ;
    +----------+
    | count(*) |
    +----------+
    | 65436672 | 
    +----------+
    1 row in set (0.00 sec)

    mysql> select count(*) from DqChannelStatusVld  ;
    +----------+
    | count(*) |
    +----------+
    |   341124 | 
    +----------+
    1 row in set (0.00 sec)

    mysql> select min(SEQNO),max(SEQNO),max(SEQNO)-min(SEQNO)+1, count(*) as N  from DqChannelStatusVld ;
    +------------+------------+-------------------------+--------+
    | min(SEQNO) | max(SEQNO) | max(SEQNO)-min(SEQNO)+1 | N      |
    +------------+------------+-------------------------+--------+
    |          1 |     341125 |                  341125 | 341124 | 
    +------------+------------+-------------------------+--------+
    1 row in set (0.00 sec)

    mysql> select min(SEQNO),max(SEQNO),max(SEQNO)-min(SEQNO)+1, count(*) as N  from DqChannelStatus ;
    +------------+------------+-------------------------+----------+
    | min(SEQNO) | max(SEQNO) | max(SEQNO)-min(SEQNO)+1 | N        |
    +------------+------------+-------------------------+----------+
    |          1 |     340817 |                  340817 | 65436672 | 
    +------------+------------+-------------------------+----------+
    1 row in set (0.01 sec)

    mysql> select 341125 -  340817 ;   /* huh 308 more validity SEQNO than payload SEQNO : DBI is not crashed payload table savvy   */
    +------------------+
    | 341125 -  340817 |
    +------------------+
    |              308 | 
    +------------------+
    1 row in set (0.03 sec)


Compare the repaired with the recreated from dump
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`tmp_ligs_offline_db_0`
              DB in which `DqChannelStatus` was repaired
`tmp_ligs_offline_db_1`
              freshly created DB populated via the mysqldump obtained from `_0` with the bad SEQNO excluded 

#. the SEQNO indicate that the Validity table continued to be updated even after the payload table had crashed


::

    mysql> select min(SEQNO),max(SEQNO),max(SEQNO)-min(SEQNO)+1, count(*) as N  from tmp_ligs_offline_db_0.DqChannelStatusVld ;
    +------------+------------+-------------------------+--------+
    | min(SEQNO) | max(SEQNO) | max(SEQNO)-min(SEQNO)+1 | N      |
    +------------+------------+-------------------------+--------+
    |          1 |     341125 |                  341125 | 341125 | 
    +------------+------------+-------------------------+--------+
    1 row in set (0.04 sec)

    mysql> select min(SEQNO),max(SEQNO),max(SEQNO)-min(SEQNO)+1, count(*) as N  from tmp_ligs_offline_db_1.DqChannelStatusVld ;
    +------------+------------+-------------------------+--------+
    | min(SEQNO) | max(SEQNO) | max(SEQNO)-min(SEQNO)+1 | N      |
    +------------+------------+-------------------------+--------+
    |          1 |     341125 |                  341125 | 341124 |    /* expected difference of 1 due to the skipped bad SEQNO */
    +------------+------------+-------------------------+--------+
    1 row in set (0.00 sec)

    mysql> select min(SEQNO),max(SEQNO),max(SEQNO)-min(SEQNO)+1, count(*) as N  from tmp_ligs_offline_db_0.DqChannelStatus ;
    +------------+------------+-------------------------+----------+
    | min(SEQNO) | max(SEQNO) | max(SEQNO)-min(SEQNO)+1 | N        |
    +------------+------------+-------------------------+----------+
    |          1 |     340817 |                  340817 | 65436731 | 
    +------------+------------+-------------------------+----------+
    1 row in set (0.05 sec)

    mysql> select min(SEQNO),max(SEQNO),max(SEQNO)-min(SEQNO)+1, count(*) as N  from tmp_ligs_offline_db_1.DqChannelStatus ;
    +------------+------------+-------------------------+----------+
    | min(SEQNO) | max(SEQNO) | max(SEQNO)-min(SEQNO)+1 | N        |
    +------------+------------+-------------------------+----------+
    |          1 |     340817 |                  340817 | 65436672 | 
    +------------+------------+-------------------------+----------+
    1 row in set (0.00 sec)

    mysql> select 65436731 -  65436672,  341125 -  340817 ;    /* the expected 59 more payloads, 308 more vld */
    +----------------------+------------------+
    | 65436731 -  65436672 | 341125 -  340817 |
    +----------------------+------------------+
    |                   59 |              308 | 
    +----------------------+------------------+
    1 row in set (0.00 sec)




Validity/Payload divergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 2-3 days of validity only writes

::

    mysql> select * from tmp_ligs_offline_db_0.DqChannelStatusVld where SEQNO in (340817,341125) ;
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | SEQNO  | TIMESTART           | TIMEEND             | SITEMASK | SIMMASK | SUBSITE | TASK | AGGREGATENO | VERSIONDATE         | INSERTDATE          |
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | 340817 | 2013-05-16 08:11:38 | 2013-05-16 08:24:05 |        2 |       1 |       1 |    0 |          -1 | 2013-05-16 08:11:38 | 2013-05-16 11:14:59 | 
    | 341125 | 2013-05-11 10:26:58 | 2013-05-11 10:43:11 |        4 |       1 |       1 |    0 |          -1 | 2013-05-11 10:26:58 | 2013-05-19 22:26:55 | 
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    2 rows in set (0.03 sec)

    mysql> select * from tmp_ligs_offline_db_1.DqChannelStatusVld where SEQNO in (340817,341125) ;
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | SEQNO  | TIMESTART           | TIMEEND             | SITEMASK | SIMMASK | SUBSITE | TASK | AGGREGATENO | VERSIONDATE         | INSERTDATE          |
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | 340817 | 2013-05-16 08:11:38 | 2013-05-16 08:24:05 |        2 |       1 |       1 |    0 |          -1 | 2013-05-16 08:11:38 | 2013-05-16 11:14:59 | 
    | 341125 | 2013-05-11 10:26:58 | 2013-05-11 10:43:11 |        4 |       1 |       1 |    0 |          -1 | 2013-05-11 10:26:58 | 2013-05-19 22:26:55 | 
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    2 rows in set (0.00 sec)


Validity only writes, 308 SEQNO 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Somehow DBI continued to write into the validity table despite the payload from be crashed and unwritable between 2013-05-16 and 2013-05-19 

::

    mysql> select * from  tmp_ligs_offline_db_0.DqChannelStatusVld where INSERTDATE > '2013-05-16 10:30:00' ;
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | SEQNO  | TIMESTART           | TIMEEND             | SITEMASK | SIMMASK | SUBSITE | TASK | AGGREGATENO | VERSIONDATE         | INSERTDATE          |
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | 340808 | 2013-05-16 08:09:49 | 2013-05-16 08:19:41 |        1 |       1 |       2 |    0 |          -1 | 2013-05-16 08:09:49 | 2013-05-16 10:30:35 | 
    | 340809 | 2013-05-16 08:09:49 | 2013-05-16 08:19:41 |        1 |       1 |       1 |    0 |          -1 | 2013-05-16 08:09:49 | 2013-05-16 10:30:37 | 
    | 340810 | 2013-05-16 07:59:53 | 2013-05-16 08:09:49 |        1 |       1 |       2 |    0 |          -1 | 2013-05-16 07:59:53 | 2013-05-16 10:41:41 | 
    | 340811 | 2013-05-16 07:59:53 | 2013-05-16 08:09:49 |        1 |       1 |       1 |    0 |          -1 | 2013-05-16 07:59:53 | 2013-05-16 10:41:43 | 
    | 340812 | 2013-05-16 07:53:39 | 2013-05-16 08:09:57 |        4 |       1 |       4 |    0 |          -1 | 2013-05-16 07:53:39 | 2013-05-16 10:48:29 | 
    | 340813 | 2013-05-16 07:53:39 | 2013-05-16 08:09:57 |        4 |       1 |       2 |    0 |          -1 | 2013-05-16 07:53:39 | 2013-05-16 10:48:31 | 
    | 340814 | 2013-05-16 07:53:39 | 2013-05-16 08:09:57 |        4 |       1 |       3 |    0 |          -1 | 2013-05-16 07:53:39 | 2013-05-16 10:48:32 | 
    | 340815 | 2013-05-16 07:53:39 | 2013-05-16 08:09:57 |        4 |       1 |       1 |    0 |          -1 | 2013-05-16 07:53:39 | 2013-05-16 10:48:35 | 
    | 340816 | 2013-05-16 08:11:38 | 2013-05-16 08:24:05 |        2 |       1 |       2 |    0 |          -1 | 2013-05-16 08:11:38 | 2013-05-16 11:14:58 | 
    | 340817 | 2013-05-16 08:11:38 | 2013-05-16 08:24:05 |        2 |       1 |       1 |    0 |          -1 | 2013-05-16 08:11:38 | 2013-05-16 11:14:59 | 
    | 340818 | 2013-05-03 03:38:35 | 2013-05-03 03:38:51 |        2 |       1 |       2 |    0 |          -1 | 2013-05-03 03:38:35 | 2013-05-19 08:22:20 |   <<< validity only SEQNO begin 
    | 340819 | 2013-05-03 03:38:35 | 2013-05-03 03:38:51 |        2 |       1 |       1 |    0 |          -1 | 2013-05-03 03:38:35 | 2013-05-19 08:22:21 | 
    | 340820 | 2013-05-08 23:49:10 | 2013-05-08 23:49:28 |        4 |       1 |       4 |    0 |          -1 | 2013-05-08 23:49:10 | 2013-05-19 08:24:37 | 
    | 340821 | 2013-05-08 23:49:10 | 2013-05-08 23:49:28 |        4 |       1 |       2 |    0 |          -1 | 2013-05-08 23:49:10 | 2013-05-19 08:24:39 | 
    | 340822 | 2013-05-08 23:49:10 | 2013-05-08 23:49:28 |        4 |       1 |       3 |    0 |          -1 | 2013-05-08 23:49:10 | 2013-05-19 08:24:40 | 
    | 340823 | 2013-05-08 23:49:10 | 2013-05-08 23:49:28 |        4 |       1 |       1 |    0 |          -1 | 2013-05-08 23:49:10 | 2013-05-19 08:24:41 | 
    | 340824 | 2013-05-03 02:11:12 | 2013-05-03 02:18:29 |        1 |       1 |       2 |    0 |          -1 | 2013-05-03 02:11:12 | 2013-05-19 09:13:33 | 
    | 340825 | 2013-05-03 02:11:12 | 2013-05-03 02:18:29 |        1 |       1 |       1 |    0 |          -1 | 2013-05-03 02:11:12 | 2013-05-19 09:13:35 | 
    | 340826 | 2013-05-09 17:37:11 | 2013-05-09 17:53:25 |        4 |       1 |       4 |    0 |          -1 | 2013-05-09 17:37:11 | 2013-05-19 09:15:57 | 
    | 340827 | 2013-05-09 17:37:11 | 2013-05-09 17:53:25 |        4 |       1 |       2 |    0 |          -1 | 2013-05-09 17:37:11 | 2013-05-19 09:15:59 | 


::

    mysql> select max(SEQNO) from DqChannelStatus ; 
    +------------+
    | max(SEQNO) |
    +------------+
    |     340817 | 
    +------------+
    1 row in set (0.00 sec)

    mysql> select * from DqChannelStatusVld where SEQNO > 340817  ;
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | SEQNO  | TIMESTART           | TIMEEND             | SITEMASK | SIMMASK | SUBSITE | TASK | AGGREGATENO | VERSIONDATE         | INSERTDATE          |
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | 340818 | 2013-05-03 03:38:35 | 2013-05-03 03:38:51 |        2 |       1 |       2 |    0 |          -1 | 2013-05-03 03:38:35 | 2013-05-19 08:22:20 | 
    | 340819 | 2013-05-03 03:38:35 | 2013-05-03 03:38:51 |        2 |       1 |       1 |    0 |          -1 | 2013-05-03 03:38:35 | 2013-05-19 08:22:21 | 
    | 340820 | 2013-05-08 23:49:10 | 2013-05-08 23:49:28 |        4 |       1 |       4 |    0 |          -1 | 2013-05-08 23:49:10 | 2013-05-19 08:24:37 | 
    | 340821 | 2013-05-08 23:49:10 | 2013-05-08 23:49:28 |        4 |       1 |       2 |    0 |          -1 | 2013-05-08 23:49:10 | 2013-05-19 08:24:39 | 
    | 340822 | 2013-05-08 23:49:10 | 2013-05-08 23:49:28 |        4 |       1 |       3 |    0 |          -1 | 2013-05-08 23:49:10 | 2013-05-19 08:24:40 | 
    | 340823 | 2013-05-08 23:49:10 | 2013-05-08 23:49:28 |        4 |       1 |       1 |    0 |          -1 | 2013-05-08 23:49:10 | 2013-05-19 08:24:41 | 
    | 340824 | 2013-05-03 02:11:12 | 2013-05-03 02:18:29 |        1 |       1 |       2 |    0 |          -1 | 2013-05-03 02:11:12 | 2013-05-19 09:13:33 | 
    ...
    | 341122 | 2013-05-11 10:26:58 | 2013-05-11 10:43:11 |        4 |       1 |       4 |    0 |          -1 | 2013-05-11 10:26:58 | 2013-05-19 22:26:30 | 
    | 341123 | 2013-05-11 10:26:58 | 2013-05-11 10:43:11 |        4 |       1 |       2 |    0 |          -1 | 2013-05-11 10:26:58 | 2013-05-19 22:26:38 | 
    | 341124 | 2013-05-11 10:26:58 | 2013-05-11 10:43:11 |        4 |       1 |       3 |    0 |          -1 | 2013-05-11 10:26:58 | 2013-05-19 22:26:47 | 
    | 341125 | 2013-05-11 10:26:58 | 2013-05-11 10:43:11 |        4 |       1 |       1 |    0 |          -1 | 2013-05-11 10:26:58 | 2013-05-19 22:26:55 | 
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    308 rows in set (0.02 sec)







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




Extraction of `DqChannel` tarball into `tmp_ligs_offline_db_0`
----------------------------------------------------------------

This is adding the IHEP `tmp_ligs_offline_db` hotcopy containing `DqChannel` tables into `tmp_ligs_offline_db_0` together with the repaired `DqChannelStatus`::


    [root@belle7 tmp_ligs_offline_db]# mysqlhotcopy.py -t 20130523_1623 --node dybdb1.ihep.ac.cn --rename tmp_ligs_offline_db_0 tmp_ligs_offline_db --ALLOWEXTRACT --ALLOWCLOBBER examine extract 
    2013-05-24 19:51:36,983 env.mysqlhotcopy.mysqlhotcopy INFO     /home/blyth/env/bin/mysqlhotcopy.py -t 20130523_1623 --node dybdb1.ihep.ac.cn --rename tmp_ligs_offline_db_0 tmp_ligs_offline_db --ALLOWEXTRACT --ALLOWCLOBBER examine extract
    2013-05-24 19:51:36,984 env.mysqlhotcopy.mysqlhotcopy INFO     backupdir /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db 
    2013-05-24 19:51:37,004 env.mysqlhotcopy.mysqlhotcopy INFO     db size in MB 0.0 
    2013-05-24 19:51:37,004 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== examine 
    2013-05-24 19:51:37,004 env.mysqlhotcopy.tar INFO     examining /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130523_1623.tar.gz 
    2013-05-24 19:51:37,004 env.mysqlhotcopy.tar WARNING  load pickled members file /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130523_1623.tar.gz.pc 
    2013-05-24 19:51:37,007 env.mysqlhotcopy.tar INFO     archive contains 7 items with commonprefix "" flattop True 
    2013-05-24 19:51:37,007 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_examine': 0.0028290748596191406, 'examine': 0.0028209686279296875} 
    2013-05-24 19:51:37,007 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== extract 
    2013-05-24 19:51:37,008 env.mysqlhotcopy.mysqlhotcopy INFO     sufficient free space,      required 0.0 MB less than    free 477552.570312 MB 
    DO YOU REALLY WANT TO extract Tar /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130523_1623.tar.gz tmp_ligs_offline_db gz  into containerdir /var/lib/mysql/    ? ENTER "YES" TO PROCEED : YES
    2013-05-24 19:51:39,842 env.mysqlhotcopy.mysqlhotcopy INFO     proceeding
    2013-05-24 19:51:39,843 env.mysqlhotcopy.mysqlhotcopy INFO     extract Tar /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130523_1623.tar.gz tmp_ligs_offline_db gz  into containerdir /var/lib/mysql/   
    2013-05-24 19:51:39,843 env.mysqlhotcopy.tar INFO     _flat_extract opening tarfile /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130523_1623.tar.gz 
    2013-05-24 19:52:51,413 env.mysqlhotcopy.tar WARNING  ./                                                                                                             :  SKIP TOPDIR 
    2013-05-24 19:52:51,413 env.mysqlhotcopy.tar INFO     extraction into target /var/lib/mysql/tmp_ligs_offline_db_0 does not clobber any existing paths 
    2013-05-24 19:52:51,413 env.mysqlhotcopy.tar INFO     _flat_extract into target /var/lib/mysql/tmp_ligs_offline_db_0 for 7 members with toplevelname tmp_ligs_offline_db_0 
    2013-05-24 19:54:04,216 env.mysqlhotcopy.tar INFO     total 6044204
    -rw-rw---- 1 mysql mysql       8892 Feb  4 16:07 DqChannel.frm
    -rw-rw---- 1 mysql mysql 2750541696 May 20 06:26 DqChannel.MYD
    -rw-rw---- 1 mysql mysql 1015181312 May 20 06:26 DqChannel.MYI
    -rw-rw---- 1 mysql mysql       8746 May 23 12:28 DqChannelStatus.frm
    -rw-rw---- 1 mysql mysql 1439608082 May 23 12:28 DqChannelStatus.MYD
    -rw-rw---- 1 mysql mysql  935564288 May 23 12:28 DqChannelStatus.MYI
    -rw-rw---- 1 mysql mysql       8908 May 13 13:16 DqChannelStatusVld.frm
    -rw-rw---- 1 mysql mysql   17397375 May 20 06:26 DqChannelStatusVld.MYD
    -rw-rw---- 1 mysql mysql    3826688 May 20 06:26 DqChannelStatusVld.MYI
    -rw-rw---- 1 mysql mysql       8908 Feb  4 16:07 DqChannelVld.frm
    -rw-rw---- 1 mysql mysql   17395539 May 20 06:26 DqChannelVld.MYD
    -rw-rw---- 1 mysql mysql    3606528 May 20 06:26 DqChannelVld.MYI

    2013-05-24 19:54:04,217 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_examine': 0.0028290748596191406, 'examine': 0.0028209686279296875, 'extract': 144.37399792671204, '_extract': 147.20948314666748} 
    [root@belle7 tmp_ligs_offline_db]# 


basic check
~~~~~~~~~~~~~

::

    mysql> use tmp_ligs_offline_db_0 

    mysql> show tables ;
    +---------------------------------+
    | Tables_in_tmp_ligs_offline_db_0 |
    +---------------------------------+
    | DqChannel                       | 
    | DqChannelStatus                 | 
    | DqChannelStatusVld              | 
    | DqChannelVld                    | 
    +---------------------------------+
    4 rows in set (0.00 sec)

    mysql> select count(*) from DqChannel ;
    +----------+
    | count(*) |
    +----------+
    | 65489088 | 
    +----------+
    1 row in set (0.00 sec)

    mysql> select count(*) from DqChannelStatus ;
    +----------+
    | count(*) |
    +----------+
    | 65436731 | 
    +----------+
    1 row in set (0.00 sec)

    mysql> select count(*) from DqChannelStatusVld ; 
    +----------+
    | count(*) |
    +----------+
    |   341125 | 
    +----------+
    1 row in set (0.00 sec)


    mysql> select count(*) from DqChannelVld ;
    +----------+
    | count(*) |
    +----------+
    |   341089 | 
    +----------+
    1 row in set (0.00 sec)



establishing correspondence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What is the criteria for establishing correspondence between DqChannel and DqChannelStatus ?


Quick and dirty
^^^^^^^^^^^^^^^^^

::

    mysql> select max(cs.seqno) from DqChannelStatusVld cs, DqChannelVld c where cs.seqno=c.seqno and cs.insertdate=c.insertdate;
    +---------------+
    | max(cs.seqno) |
    +---------------+
    |        323573 |
    +---------------+
    1 row in set (1.64 sec)

This query indicates when the synchronized writing
starts to go a long way astray but it is
not a reliable technique due to flawed assumptions.

* same second inserts to two tables
* SEQNO correspondence between two tables


Better but slower way
^^^^^^^^^^^^^^^^^^^^^^^^^^

Based on run range comparisons of "group by SEQNO" queries for each and comparing the RUNNO/FILENO
::

    mysql> select SEQNO, count(*) as N, RUNNO, FILENO from DqChannelStatus group by SEQNO limit 10 ;
    +-------+-----+-------+--------+
    | SEQNO | N   | RUNNO | FILENO |
    +-------+-----+-------+--------+
    |     1 | 192 | 21223 |      1 | 
    |     2 | 192 | 21223 |      1 | 
    |     3 | 192 | 21223 |      1 | 
    |     4 | 192 | 37322 |    442 | 
    |     5 | 192 | 37322 |    442 | 
    |     6 | 192 | 37322 |    441 | 
    |     7 | 192 | 37322 |    441 | 
    |     8 | 192 | 37325 |    351 | 
    |     9 | 192 | 37325 |    351 | 
    |    10 | 192 | 37325 |    352 | 
    +-------+-----+-------+--------+
    10 rows in set (0.01 sec)

    mysql> select SEQNO, count(*) as N, RUNNO, FILENO from DqChannel group by SEQNO limit 10 ;
    +-------+-----+-------+--------+
    | SEQNO | N   | RUNNO | FILENO |
    +-------+-----+-------+--------+
    |     1 | 192 | 21223 |      1 | 
    |     2 | 192 | 21223 |      1 | 
    |     3 | 192 | 21223 |      1 | 
    |     4 | 192 | 37322 |    442 | 
    |     5 | 192 | 37322 |    442 | 
    |     6 | 192 | 37322 |    441 | 
    |     7 | 192 | 37322 |    441 | 
    |     8 | 192 | 37325 |    351 | 
    |     9 | 192 | 37325 |    351 | 
    |    10 | 192 | 37325 |    352 | 
    +-------+-----+-------+--------+
    10 rows in set (0.01 sec)


I checked correspondence between  DqChannel and the repaired DqChannelStatus in `tmp_ligs_offline_db_0` at NUU.

http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Database/Scraper/python/Scraper/dq/cq_zip_check.py

Many ordering swaps are apparent.

Presumably the explanation of this is that multiple instances of the filling script
are closing ingredients and summary writers concurrently.
This breaks the sequentiality of closing of the two writers
from any one instance of your script preventing them having the
same SEQNO in the two tables (at least not reliably).

If sequential KUP job running is not possible then
in order to make syncronized SEQNO writing to two tables
you will need to try wrapping the closing in lock/unlock.
Something like::

         db("lock tables DqChannel WRITE, DqChannelVld WRITE, DqChannelStatus WRITE, DqChannelStatusVld WRITE")
         wseqno = wrt.Close()
         wseqno_status = wrt_status.Close()
         db("unlock tables")
         assert wseqno ==  wseqno_status

In this way the first instance of the script to take the lock will be able
to sequentially perform its writes before releasing its lock.  Other scripts
will hang around until the first is done and so on.

This should allow synchronized writing in future, but does not
fix the existing lack of synchronized nature in the tables so far.
I will prepare a dump with the "SEQNO <= 323573" cut to allow you to
check out my observations.


Did this with :dybsvn:`source:dybgaudi/trunk/Database/Scraper/python/Scraper/dq/cq_zip_check.py`


Concurrent DBI Writing
------------------------

Some small DBI mods allow to disable the DBI locking and this together with 
another trick to use a single session gives controlled concurrent writing.

* :dybsvn:`changeset:20618`
* :dybsvn:`changeset:20619`
* :dybsvn:`changeset:20620`

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Database/DybDbi/tests/test_dbi_locking.sh

Most of the time this works providing controlled concurrent writing with external locking. 
But there is enough concurrent flakiness (maybe 1 out of 5 runs of the above test) 
that result in failed writes that it cannot be recommended at the moment.  

The case for synced DBI writing to multiple tables is 
not strong enough to merit much more work on this.


Four Table Dump/load, 8/70 min 
-------------------------------

mysqldump are fast to dump (8 min), but very slow to load  (70 min)

* possibly load options can be tweaked to go faster
* or alternate dump technique used

::

    [blyth@belle7 DybPython]$ dbdumpload.py tmp_ligs_offline_db_0 dump ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql --where 'SEQNO <= 323573' --tables 'DqChannelStatus DqChannelStatusVld DqChannel DqChannelVld'  
    [blyth@belle7 DybPython]$ dbdumpload.py tmp_ligs_offline_db_0 dump ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql --where 'SEQNO <= 323573' --tables 'DqChannelStatus DqChannelStatusVld DqChannel DqChannelVld'  | sh 

    real    8m37.035s
    user    3m3.306s
    sys     0m23.131s
    [blyth@belle7 DybPython]$ du -h  ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql
    5.7G    /home/blyth/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql

    [blyth@belle7 DybPython]$ tail -c 1000  ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql
    [blyth@belle7 DybPython]$ head -c 1000  ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql
    [blyth@belle7 DybPython]$ grep CREATE ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql
    CREATE TABLE `DqChannelStatus` (
    CREATE TABLE `DqChannelStatusVld` (
    CREATE TABLE `DqChannel` (
    CREATE TABLE `DqChannelVld` (
    [blyth@belle7 DybPython]$ grep DROP ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql
    [blyth@belle7 DybPython]$ md5sum ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql
    ea8a5a4d076febbfd940a90171707a72  /home/blyth/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql


Alternative dump/load technique, 8/32 min  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://dev.mysql.com/doc/refman/5.0/en/insert-speed.html

::

    blyth@belle7 DybPython]$ time ./dbsrv.py  tmp_ligs_offline_db_0 dumplocal ~/tmp_ligs_offline_db_0 --where 'SEQNO <= 323573' -l debug 
    DEBUG:__main__:MyCnf read ['/home/blyth/.my.cnf'] 
    DEBUG:__main__:translate mysql config {'host': 'belle7.nuu.edu.tw', 'password': '***', 'user': 'root', 'database': 'tmp_ligs_offline_db_0'} into mysql-python config {'passwd': '***', 'host': 'belle7.nuu.edu.tw', 'db': 'tmp_ligs_offline_db_0', 'user': 'root'} 
    DEBUG:__main__:connecting to {'passwd': '***', 'host': 'belle7.nuu.edu.tw', 'db': 'tmp_ligs_offline_db_0', 'user': 'root'} 
    DEBUG:__main__:select distinct(table_name) from information_schema.tables where table_schema='tmp_ligs_offline_db_0'
    DEBUG:__main__:show create table DqChannel
    DEBUG:__main__:select * from DqChannel where SEQNO <= 323573 into outfile '/home/blyth/tmp_ligs_offline_db_0/DqChannel.csv' fields terminated by ',' optionally enclosed by '"' 
    DEBUG:__main__:show create table DqChannelStatus
    DEBUG:__main__:select * from DqChannelStatus where SEQNO <= 323573 into outfile '/home/blyth/tmp_ligs_offline_db_0/DqChannelStatus.csv' fields terminated by ',' optionally enclosed by '"' 
    DEBUG:__main__:show create table DqChannelStatusVld
    DEBUG:__main__:select * from DqChannelStatusVld where SEQNO <= 323573 into outfile '/home/blyth/tmp_ligs_offline_db_0/DqChannelStatusVld.csv' fields terminated by ',' optionally enclosed by '"' 
    DEBUG:__main__:show create table DqChannelVld
    DEBUG:__main__:select * from DqChannelVld where SEQNO <= 323573 into outfile '/home/blyth/tmp_ligs_offline_db_0/DqChannelVld.csv' fields terminated by ',' optionally enclosed by '"' 

    real    8m11.323s
    user    0m0.269s
    sys     0m0.087s
    [blyth@belle7 DybPython]$ 


::

    [blyth@belle7 DybPython]$ time ./dbsrv.py tmp_ligs_offline_db_4 loadlocal ~/tmp_ligs_offline_db_0  -l debug --DB_DROP_CREATE -C
    ...
      PRIMARY KEY  (`SEQNO`)
    ) ENGINE=MyISAM AUTO_INCREMENT=341090 DEFAULT CHARSET=latin1
    DEBUG:__main__:LOAD DATA LOCAL INFILE '/home/blyth/tmp_ligs_offline_db_0/DqChannelVld.csv' IGNORE INTO TABLE DqChannelVld FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' IGNORE 0 LINES 

    real    32m38.231s
    user    0m1.639s
    sys     0m6.183s
    [blyth@belle7 DybPython]$ 


    [blyth@belle7 DybPython]$ ./dbsrv.py tmp_ligs_offline_db_4 summary                                                        
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TABLE_NAME                      TABLE_ROWS  CREATE_TIME                     CHECK_TIME                    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DqChannel                       62126016    2013-05-30 13:54:33             2013-05-30 14:11:53           
    DqChannelStatus                 62126016    2013-05-30 14:11:54             2013-05-30 14:26:55           
    DqChannelStatusVld              323573      2013-05-30 14:26:56             None                          
    DqChannelVld                    323573      2013-05-30 14:26:58             None                          
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




Fake LOCALSEQNO
----------------

::

    [blyth@belle7 ~]$ path=~/LOCALSEQNO.sql 
    [blyth@belle7 ~]$ dbdumpload.py -t LOCALSEQNO --no-data tmp_offline_db dump $path | sh 
    [blyth@belle7 ~]$ maxseqno=323573 
    [blyth@belle7 ~]$ echo "INSERT INTO LOCALSEQNO VALUES ('*',0),('DqChannel',$maxseqno),('DqChannelStatus',$maxseqno);" >> $path
    [blyth@belle7 ~]$ echo drop database if exists test_localseqno | mysql 
    [blyth@belle7 ~]$ echo create database test_localseqno | mysql 
    [blyth@belle7 ~]$ cat $path | mysql test_localseqno 
    [blyth@belle7 ~]$ echo select \* from LOCALSEQNO | mysql test_localseqno -t
    +-----------------+---------------+
    | TABLENAME       | LASTUSEDSEQNO |
    +-----------------+---------------+
    | *               |             0 | 
    | DqChannel       |        323573 | 
    | DqChannelStatus |        323573 | 
    +-----------------+---------------+


Append LOCALSEQNO onto the dump 
---------------------------------

::

    [blyth@belle7 ~]$ cat $path >> ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql 
    [blyth@belle7 ~]$ du -hs ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql 
    5.7G    /home/blyth/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql
    [blyth@belle7 ~]$ md5sum ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql 
    8aed64440efb14d3676b8fda1bc85e5e  /home/blyth/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql


Create DB `_2` from the four table dump with faked LOCALSEQNO
----------------------------------------------------------------

::

    [blyth@belle7 ~]$ db=tmp_ligs_offline_db_2
    [blyth@belle7 ~]$ echo drop database if exists $db | mysql 
    [blyth@belle7 ~]$ echo create database $db | mysql 
    [blyth@belle7 ~]$ time cat ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql | mysql $db
    real    72m18.139s
    user    3m0.786s
    sys     0m24.214s


* OUCH: 72 min to load the dump, this is liable to kill the server for other users 


.. warning:: disk space usage from the cat could easily be more than 3 times the size of the dump due to the new DB and mysql logging


try alternative load to check time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Almost same time as piped cat::

    [blyth@belle7 ~]$ db=tmp_ligs_offline_db_3 && echo drop database if exists $db | mysql && echo create database $db | mysql 
    [blyth@belle7 ~]$ time mysql $db < ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql  
    real    72m24.332s
    user    2m44.720s
    sys     0m13.221s
    [blyth@belle7 ~]$ 



loading alternatives
~~~~~~~~~~~~~~~~~~~~~~

#. slow mysqldump
#. csv style `forced_rloadcat` with `--local` on server thus used the fast `LOAD DATA LOCAL INFILE`

#. mysqlhotcopy.py archive and extract

   * its really fast 
   * BUT: concern about mysql version differnce between table creation server and table repair sever 



dump alternatives
~~~~~~~~~~~~~~~~~~~~

::

    mysql> show tables ;
    +---------------------------------+
    | Tables_in_tmp_ligs_offline_db_0 |
    +---------------------------------+
    | DqChannel                       | 
    | DqChannelStatus                 | 
    | DqChannelStatusVld              | 
    | DqChannelVld                    | 
    +---------------------------------+
    4 rows in set (0.00 sec)

    mysql> select * from DqChannel where SEQNO < 100 into outfile '/tmp/DqChannel.csv' fields terminated by ',' optionally enclosed by '"' ;
    Query OK, 19008 rows affected (0.38 sec)




Deciding how to proceed
--------------------------

I have a recovery dump file for tmp_ligs_offline_db, 
however load times are too long to be used on 
the primary server.

* 70 min : from mysqldump
* 35 min : from CSV based data with "LOAD DATA LOCAL INFILE" 

An alternative would be to extract a "mysqlhotcopy" tarball 
created elsewhere onto dybdb1.ihep.ac.cn.  
That would probably take less than 10 min and it does not impose 
such a high load on the server.
  
I could make the hotcopy on belle7 (server version 5.0.77) 
and archive it into a tarball to be extracted on dybdb1.ihep.ac.cn
(server version  5.0.45). But that might cause problems in 
future as creating tables on a version of MySQL different 
from the version on which you might in future need to make repairs 
limits repair techniques that can be used.

      http://dev.mysql.com/doc/refman/5.0/en/repair-table.html

(The recent incident required repairing elsewhere as we had no 
available backup in hand and you never want to attempt a repair 
without having an available and verified backup.)


* decide to install MySQL 5.0.45 RPM on DB virgin belle1



MySQL Server versions and repair table limitation
---------------------------------------------------

Server versions, our primary servers use ``5.0.45``

    =======================  ========================
     node                         server version
    =======================  ========================
     dybdb1.ihep.ac.cn        5.0.45
     dybdb2.ihep.ac.cn        5.0.45
     dayabay.ihep.ac.cn       5.1.36
     belle7.nuu.edu.tw        5.0.77 
     belle1.nuu.edu.tw        5.0.45 see `mysqlrpm-` 
     cms01.phys.ntu.edu.tw    4.1.22
    =======================  ========================
  
version shifting repair issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://dev.mysql.com/doc/refman/5.0/en/repair-table.html

Prior to MySQL 5.0.62, do not use USE_FRM if your table was created by a
different version of the MySQL server. Doing so risks the loss of all rows in
the table. It is particularly dangerous to use USE_FRM after the server returns
this message::

    Table upgrade required. Please do
    "REPAIR TABLE `tbl_name`" to fix it!

Does **different version of the MySQL server** refer to major or minor versions ?

what this means
~~~~~~~~~~~~~~~~~~

It is better for tables to be created on the same server version as they are 
used and potentially repaired. Thus install 5.0.45 from RPM on belle1 in 
order to be able to create a same version hotcopy for extraction into dybdb1.
See `mysqlrpm-` for the install sage.


Compare MySQL servers on belle1 and dybdb1
-------------------------------------------

dybddb1
~~~~~~~~

Remote connection to dybdb1 from belle7::

    mysql> status ;
    --------------
    /data1/env/local/dyb/external/mysql/5.0.67/i686-slc5-gcc41-dbg/bin/mysql  Ver 14.12 Distrib 5.0.67, for redhat-linux-gnu (i686) using  EditLine wrapper

    Connection id:          610209
    Current database:       tmp_ligs_offline_db
    Current user:           ligs@belle7.nuu.edu.tw
    SSL:                    Not in use
    Current pager:          stdout
    Using outfile:          ''
    Using delimiter:        ;
    Server version:         5.0.45-community-log MySQL Community Edition (GPL)
    Protocol version:       10
    Connection:             dybdb1.ihep.ac.cn via TCP/IP
    Server characterset:    latin1
    Db     characterset:    latin1
    Client characterset:    latin1
    Conn.  characterset:    latin1
    TCP port:               3306
    Uptime:                 12 days 6 hours 51 min 8 sec

    Threads: 8  Questions: 171104994  Slow queries: 79  Opens: 335  Flush tables: 1  Open tables: 302  Queries per second avg: 161.197
    --------------


::

    mysql>  select table_schema, table_name, table_collation from information_schema.tables where table_schema = 'tmp_ligs_offline_db' ;
    +---------------------+-----------------------+-------------------+
    | table_schema        | table_name            | table_collation   |
    +---------------------+-----------------------+-------------------+
    | tmp_ligs_offline_db | ChannelQuality        | latin1_swedish_ci | 
    | tmp_ligs_offline_db | ChannelQualityVld     | latin1_swedish_ci | 
    | tmp_ligs_offline_db | DaqRawDataFileInfo    | latin1_swedish_ci | 
    | tmp_ligs_offline_db | DaqRawDataFileInfoVld | latin1_swedish_ci | 
    | tmp_ligs_offline_db | DqChannel             | latin1_swedish_ci | 
    | tmp_ligs_offline_db | DqChannelStatus       | NULL              | 
    | tmp_ligs_offline_db | DqChannelStatusVld    | latin1_swedish_ci | 
    | tmp_ligs_offline_db | DqChannelVld          | latin1_swedish_ci | 
    | tmp_ligs_offline_db | LOCALSEQNO            | latin1_swedish_ci | 
    +---------------------+-----------------------+-------------------+
    9 rows in set (0.07 sec)


belle1
~~~~~~~

Local connection to belle1::

    mysql> status 
    --------------
    mysql  Ver 14.12 Distrib 5.0.45, for pc-linux-gnu (i686) using readline 5.0

    Connection id:          28
    Current database:       information_schema
    Current user:           root@localhost
    SSL:                    Not in use
    Current pager:          stdout
    Using outfile:          ''
    Using delimiter:        ;
    Server version:         5.0.45-community MySQL Community Edition (GPL)
    Protocol version:       10
    Connection:             127.0.0.1 via TCP/IP
    Server characterset:    latin1
    Db     characterset:    utf8
    Client characterset:    latin1
    Conn.  characterset:    latin1
    TCP port:               3306
    Uptime:                 50 min 57 sec

    Threads: 2  Questions: 114  Slow queries: 0  Opens: 23  Flush tables: 1  Open tables: 17  Queries per second avg: 0.037



Only difference is Db characterset

* http://dev.mysql.com/doc/refman/5.0/en/charset-database.html


::

    mysql> select @@character_set_database ;
    +--------------------------+
    | @@character_set_database |
    +--------------------------+
    | utf8                     | 
    +--------------------------+
    1 row in set (0.00 sec)


The character set and collation for the default database can be determined from
the values of the character_set_database and collation_database system
variables. The server sets these variables whenever the default database
changes. If there is no default database, the variables have the same value as
the corresponding server-level system variables, character_set_server and
collation_server.

::

    mysql> select table_name, table_collation from tables where table_schema = 'channelquality_db' ;
    +--------------------+-------------------+
    | table_name         | table_collation   |
    +--------------------+-------------------+
    | DqChannel          | latin1_swedish_ci | 
    | DqChannelStatus    | latin1_swedish_ci | 
    | DqChannelStatusVld | latin1_swedish_ci | 
    | DqChannelVld       | latin1_swedish_ci | 
    | LOCALSEQNO         | latin1_swedish_ci | 
    +--------------------+-------------------+
    5 rows in set (0.00 sec)




Load the dump into belle1
---------------------------

::

    [blyth@belle1 ~]$ md5sum tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql
    8aed64440efb14d3676b8fda1bc85e5e  tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql
    8aed64440efb14d3676b8fda1bc85e5e   
    [blyth@belle1 ~]$ echo 8aed64440efb14d3676b8fda1bc85e5e    # matches digest from belle7
    [blyth@belle1 ~]$ 
    [blyth@belle1 ~]$ echo create database channelquality_db | mysql 
    [blyth@belle1 ~]$ time mysql channelquality_db < ~/tmp_ligs_offline_db_0.DqChannel_and_DqChannelStatus.sql 
    real    77m19.981s
    user    2m45.547s
    sys     0m12.736s
    [blyth@belle1 ~]$ 


Checking the load as it progresses::

    mysql> select TABLE_NAME, TABLE_TYPE, ENGINE, TABLE_ROWS, CREATE_TIME, UPDATE_TIME from information_schema.tables where table_schema = 'channelquality_db' ;
    +--------------------+------------+--------+------------+---------------------+---------------------+
    | TABLE_NAME         | TABLE_TYPE | ENGINE | TABLE_ROWS | CREATE_TIME         | UPDATE_TIME         |
    +--------------------+------------+--------+------------+---------------------+---------------------+
    | DqChannel          | BASE TABLE | MyISAM |   59651813 | 2013-05-30 18:52:51 | 2013-05-30 19:33:07 | 
    | DqChannelStatus    | BASE TABLE | MyISAM |   62126016 | 2013-05-30 18:17:42 | 2013-05-30 18:52:44 | 
    | DqChannelStatusVld | BASE TABLE | MyISAM |     323573 | 2013-05-30 18:52:44 | 2013-05-30 18:52:51 | 
    +--------------------+------------+--------+------------+---------------------+---------------------+
    3 rows in set (0.00 sec)

At completion::

    mysql> select TABLE_NAME, TABLE_TYPE, ENGINE, TABLE_ROWS, CREATE_TIME, UPDATE_TIME from information_schema.tables where table_schema = 'channelquality_db' ;
    +--------------------+------------+--------+------------+---------------------+---------------------+
    | TABLE_NAME         | TABLE_TYPE | ENGINE | TABLE_ROWS | CREATE_TIME         | UPDATE_TIME         |
    +--------------------+------------+--------+------------+---------------------+---------------------+
    | DqChannel          | BASE TABLE | MyISAM |   62126016 | 2013-05-30 18:52:51 | 2013-05-30 19:34:55 | 
    | DqChannelStatus    | BASE TABLE | MyISAM |   62126016 | 2013-05-30 18:17:42 | 2013-05-30 18:52:44 | 
    | DqChannelStatusVld | BASE TABLE | MyISAM |     323573 | 2013-05-30 18:52:44 | 2013-05-30 18:52:51 | 
    | DqChannelVld       | BASE TABLE | MyISAM |     323573 | 2013-05-30 19:34:55 | 2013-05-30 19:35:02 | 
    | LOCALSEQNO         | BASE TABLE | MyISAM |          3 | 2013-05-30 19:35:02 | 2013-05-30 19:35:02 | 
    +--------------------+------------+--------+------------+---------------------+---------------------+
    5 rows in set (0.00 sec)



belle1 hotcopy
---------------

After dealing with a mysqlhotcopy perl issue, `mysqlrpm-`

::

    [root@belle1 ~]#  mysqlhotcopy.py -l debug channelquality_db hotcopy archive
    2013-05-30 20:29:40,578 env.mysqlhotcopy.mysqlhotcopy INFO     /home/blyth/env/bin/mysqlhotcopy.py -l debug channelquality_db hotcopy archive
    2013-05-30 20:29:40,582 env.mysqlhotcopy.mysqlhotcopy INFO     backupdir /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db 
    2013-05-30 20:29:40,582 env.mysqlhotcopy.db DEBUG    MyCnf read ['/root/.my.cnf'] 
    2013-05-30 20:29:40,582 env.mysqlhotcopy.db DEBUG    translate mysql config {'host': 'localhost', 'user': 'root', 'database': 'information_schema', 'password': '***', 'socket': '/var/lib/mysql/mysql.sock'} into mysql-python config {'unix_socket': '/var/lib/mysql/mysql.sock', 'host': 'localhost', 'user': 'root', 'passwd': '***', 'db': 'information_schema'} 
    2013-05-30 20:29:40,582 env.mysqlhotcopy.db DEBUG    connecting to {'unix_socket': '/var/lib/mysql/mysql.sock', 'host': 'localhost', 'user': 'root', 'passwd': '***', 'db': 'information_schema'} 
    2013-05-30 20:29:40,583 env.mysqlhotcopy.mysqlhotcopy INFO     failed to instanciate connection to database channelquality_db with exception 'NoneType' object has no attribute 'Error' 
    2013-05-30 20:29:40,583 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== hotcopy 
    2013-05-30 20:29:40,583 env.mysqlhotcopy.mysqlhotcopy WARNING  no valid db connection using static opts.mb_required 2000 
    2013-05-30 20:29:40,583 env.mysqlhotcopy.mysqlhotcopy INFO     sufficient free space,      required 2000 MB less than    free 72771.5898438 MB 
    2013-05-30 20:29:40,583 env.mysqlhotcopy.mysqlhotcopy INFO     hotcopy of database channelquality_db into outd /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029 
    2013-05-30 20:29:40,586 env.mysqlhotcopy.mysqlhotcopy INFO     proceed with MySQLHotCopy /usr/bin/mysqlhotcopy  channelquality_db /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029   
    2013-05-30 20:29:40,586 env.mysqlhotcopy.cmd DEBUG    MySQLHotCopy /usr/bin/mysqlhotcopy  channelquality_db /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029  
    2013-05-30 20:34:38,323 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_hotcopy': 297.73979902267456} 
    2013-05-30 20:34:38,323 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== archive 
    2013-05-30 20:34:38,324 env.mysqlhotcopy.mysqlhotcopy WARNING  no valid db connection using static opts.mb_required 2000 
    2013-05-30 20:34:38,324 env.mysqlhotcopy.mysqlhotcopy INFO     sufficient free space,      required 2000 MB less than    free 63394.0234375 MB 
    2013-05-30 20:34:38,324 env.mysqlhotcopy.mysqlhotcopy INFO     tagd /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029  into Tar /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz channelquality_db gz  
    2013-05-30 20:34:38,324 env.mysqlhotcopy.tar INFO     creating /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz from /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db 



The hotcopy step only took 5min for 9 GB of hotcopied directory:: 


    [root@belle1 ~]# du -hs /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/
    9.2G    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/
    [root@belle1 ~]# 
    [root@belle1 ~]# du -hs /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/*
    4.0K    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/db.opt
    12K     /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannel.frm
    2.5G    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannel.MYD
    2.8G    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannel.MYI
    12K     /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannelStatus.frm
    1.3G    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannelStatus.MYD
    2.8G    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannelStatus.MYI
    12K     /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannelStatusVld.frm
    16M     /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannelStatusVld.MYD
    3.5M    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannelStatusVld.MYI
    12K     /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannelVld.frm
    16M     /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannelVld.MYD
    3.3M    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/DqChannelVld.MYI
    12K     /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/LOCALSEQNO.frm
    4.0K    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/LOCALSEQNO.MYD
    4.0K    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db/LOCALSEQNO.MYI
    [root@belle1 ~]# 



Compressing this into archive is too slow::

    [root@belle1 ~]# du -h /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz 
    479M    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz
    [root@belle1 ~]# 


Did it from cron::

    [root@belle1 ~]# crontab -l
    SHELL = /bin/bash
    PATH=/home/blyth/env/bin:/usr/bin:/bin
    04 21 * * * ( mysqlhotcopy.py -l debug -t 20130530_2029 channelquality_db archive > /root/mysqlhotcopy.log 2>&1 )
    [root@belle1 ~]# 

Archiving a 9.2G directory down to 2.3G tarball took 4 hrs, unimportant error from forgetting no-confirm option for sourcedir deletion::

    [root@belle1 ~]# cat mysqlhotcopy.log 
    2013-05-30 21:04:01,229 env.mysqlhotcopy.mysqlhotcopy INFO     /home/blyth/env/bin/mysqlhotcopy.py -l debug -t 20130530_2029 channelquality_db archive
    2013-05-30 21:04:01,232 env.mysqlhotcopy.mysqlhotcopy INFO     backupdir /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db 
    2013-05-30 21:04:01,233 env.mysqlhotcopy.db DEBUG    MyCnf read ['/root/.my.cnf'] 
    2013-05-30 21:04:01,233 env.mysqlhotcopy.db DEBUG    translate mysql config {'host': 'localhost', 'user': 'root', 'database': 'information_schema', 'password': '***', 'socket': '/var/lib/mysql/mysql.sock'} into mysql-python config {'unix_socket': '/var/lib/mysql/mysql.sock', 'host': 'localhost', 'user': 'root', 'passwd': '***', 'db': 'information_schema'} 
    2013-05-30 21:04:01,233 env.mysqlhotcopy.db DEBUG    connecting to {'unix_socket': '/var/lib/mysql/mysql.sock', 'host': 'localhost', 'user': 'root', 'passwd': '***', 'db': 'information_schema'} 
    2013-05-30 21:04:01,233 env.mysqlhotcopy.mysqlhotcopy INFO     failed to instanciate connection to database channelquality_db with exception 'NoneType' object has no attribute 'Error' 
    2013-05-30 21:04:01,234 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== archive 
    2013-05-30 21:04:01,234 env.mysqlhotcopy.mysqlhotcopy WARNING  no valid db connection using static opts.mb_required 2000 
    2013-05-30 21:04:01,234 env.mysqlhotcopy.mysqlhotcopy INFO     sufficient free space,      required 2000 MB less than    free 63394.015625 MB 
    2013-05-30 21:04:01,234 env.mysqlhotcopy.mysqlhotcopy INFO     tagd /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029  into Tar /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz channelquality_db gz  
    2013-05-30 21:04:01,234 env.mysqlhotcopy.tar INFO     creating /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz from /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/channelquality_db 
    2013-05-31 00:59:05,021 env.mysqlhotcopy.tar INFO     deleting sourcedir /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029 with leaf 20130530_2029 as the leaf is a dated folder 
    enter "YES" to confirm deletion of sourcedir /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029 :Traceback (most recent call last):
      File "/home/blyth/env/bin/mysqlhotcopy.py", line 4, in ?
        main()
      File "/usr/lib/python2.4/site-packages/env/mysqlhotcopy/mysqlhotcopy.py", line 721, in main
        hb(verb)
      File "/usr/lib/python2.4/site-packages/env/mysqlhotcopy/mysqlhotcopy.py", line 470, in __call__
        self._archive()
      File "/usr/lib/python2.4/site-packages/env/mysqlhotcopy/common.py", line 13, in wrapper
        res = func(*arg,**kw)
      File "/usr/lib/python2.4/site-packages/env/mysqlhotcopy/mysqlhotcopy.py", line 570, in _archive
        self.tar.archive(self.tagd, self.opts.deleteafter, self.opts.flattop) 
      File "/usr/lib/python2.4/site-packages/env/mysqlhotcopy/common.py", line 13, in wrapper
        res = func(*arg,**kw)
      File "/usr/lib/python2.4/site-packages/env/mysqlhotcopy/tar.py", line 155, in archive
        confirm = raw_input("enter \"YES\" to confirm deletion of sourcedir %s :" % sourcedir )
    EOFError: EOF when reading a line
    [root@belle1 ~]# 
    [root@belle1 ~]# 
    [root@belle1 ~]# du -hs /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/
    9.2G    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029/

    [root@belle1 ~]# du -h /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz
    2.3G    /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz

    [root@belle1 ~]# tar ztvf  /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz
    drwxr-x--- mysql/mysql       0 2013-05-30 20:34:38 channelquality_db/
    -rw-rw---- mysql/mysql    8618 2013-05-30 19:35:02 channelquality_db/LOCALSEQNO.frm
    -rw-rw---- mysql/mysql 3646464 2013-05-30 20:29:40 channelquality_db/DqChannelStatusVld.MYI
    -rw-rw---- mysql/mysql    8746 2013-05-30 18:17:42 channelquality_db/DqChannelStatus.frm
    -rw-rw---- mysql/mysql 2609292672 2013-05-30 19:34:55 channelquality_db/DqChannel.MYD
    -rw-rw---- mysql/mysql 2901941248 2013-05-30 20:29:40 channelquality_db/DqChannel.MYI
    -rw-rw---- mysql/mysql         65 2013-05-30 18:17:10 channelquality_db/db.opt
    -rw-rw---- mysql/mysql       2048 2013-05-30 20:29:40 channelquality_db/LOCALSEQNO.MYI
    -rw-rw---- mysql/mysql 2905288704 2013-05-30 20:29:40 channelquality_db/DqChannelStatus.MYI
    -rw-rw---- mysql/mysql 1366772352 2013-05-30 18:52:44 channelquality_db/DqChannelStatus.MYD
    -rw-rw---- mysql/mysql   16502223 2013-05-30 18:52:51 channelquality_db/DqChannelStatusVld.MYD
    -rw-rw---- mysql/mysql       8908 2013-05-30 19:34:55 channelquality_db/DqChannelVld.frm
    -rw-rw---- mysql/mysql       8892 2013-05-30 18:52:51 channelquality_db/DqChannel.frm
    -rw-rw---- mysql/mysql   16502223 2013-05-30 19:35:02 channelquality_db/DqChannelVld.MYD
    -rw-rw---- mysql/mysql        207 2013-05-30 19:35:02 channelquality_db/LOCALSEQNO.MYD
    -rw-rw---- mysql/mysql       8908 2013-05-30 18:52:44 channelquality_db/DqChannelStatusVld.frm
    -rw-rw---- mysql/mysql    3427328 2013-05-30 20:29:40 channelquality_db/DqChannelVld.MYI
    [root@belle1 ~]# 
    [root@belle1 ~]# 
    [root@belle1 ~]# 


OOPS didnt use `--flattop`. Takes too long to rerun for this though. 
Actually that simplifies manual extraction, but makes database renaming problematic.
As the name is already as desired "channelquality_db" thats no problem.


Test extraction of belle1 hotcopy onto belle7
-----------------------------------------------

Prepare directory for the tarball on belle7 and scp it over from belle1, taking 3.5 min::

    [root@belle7 ~]# mkdir -p  /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/
    [root@belle7 ~]# time scp N1:/var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/
    real    3m28.167s
    user    1m19.160s
    sys     0m24.959s

Verify the digests match::

    [root@belle7 ~]# ssh N1 md5sum /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz
    2631bcc9b9c747e238338a4b50c04ad5  /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz
    [root@belle7 ~]# md5sum /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz
    2631bcc9b9c747e238338a4b50c04ad5  /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz

Check to see what the mysql datadir is::

    [root@belle7 ~]# vim .my.cnf   # check the "client" section is appropriate 
    [root@belle7 ~]# echo select \@\@datadir | mysql  
    @@datadir
    /var/lib/mysql/

Extract into belle7 datadir, took less than 5 min to extract out to 9.2 G::

    [root@belle7 ~]# cd /var/lib/mysql
    [root@belle7 mysql]# time tar zxvf /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz
    channelquality_db/
    channelquality_db/LOCALSEQNO.frm
    channelquality_db/DqChannelStatusVld.MYI
    channelquality_db/DqChannelStatus.frm
    channelquality_db/DqChannel.MYD
    channelquality_db/DqChannel.MYI
    channelquality_db/db.opt
    channelquality_db/LOCALSEQNO.MYI
    channelquality_db/DqChannelStatus.MYI
    channelquality_db/DqChannelStatus.MYD
    channelquality_db/DqChannelStatusVld.MYD
    channelquality_db/DqChannelVld.frm
    channelquality_db/DqChannel.frm
    channelquality_db/DqChannelVld.MYD
    channelquality_db/LOCALSEQNO.MYD
    channelquality_db/DqChannelStatusVld.frm
    channelquality_db/DqChannelVld.MYI

    real    4m30.838s
    user    1m34.536s
    sys     0m40.571s
    [root@belle7 mysql]# 
    [root@belle7 mysql]# du -hs channelquality_db
    9.2G    channelquality_db


basic checks OK
~~~~~~~~~~~~~~~~~~

::

    mysql> use channelquality_db 
    Database changed
    mysql> show tables ;
    +-----------------------------+
    | Tables_in_channelquality_db |
    +-----------------------------+
    | DqChannel                   | 
    | DqChannelStatus             | 
    | DqChannelStatusVld          | 
    | DqChannelVld                | 
    | LOCALSEQNO                  | 
    +-----------------------------+
    5 rows in set (0.00 sec)

    mysql> select max(SEQNO) from  DqChannel ;
    +------------+
    | max(SEQNO) |
    +------------+
    |     323573 | 
    +------------+
    1 row in set (0.01 sec)


place tarball on cms01 where Qiumei has access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare directory on S::

    [root@cms01 ~]# mkdir -p /data/var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/
    [root@cms01 ~]# chown -R dayabayscp.dayabayscp  /data/var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/

Go around the NUU-NTU blockade via my laptop::

    simon:~ blyth$ scp N:/var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz .
    simon:~ blyth$ scp 20130530_2029.tar.gz S:/data/var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz
    simon:~ blyth$ ssh S md5sum /data/var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz          ## nope the dayabayscp user has restricted shell
    simon:~ blyth$ ssh C md5sum /data/var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz
    2631bcc9b9c747e238338a4b50c04ad5  /data/var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz


dybdb1 extraction instructions for Qiumei
-------------------------------------------



.. sidebar:: how hotcopy tarball was prepared

     On belle1, I installed MySQL-server 5.0.45 precisely matching the version on dybdb1.  
     This is to avoid potential repair limitations in future. Also tarball extraction of the 
     prepared DB is the approach that minimizes load on dybdb1. Alternatives like loading 
     mysqldumps or CSVs were found to make the server unresponsive for long periods, 30-60 mins.

.. contents:: :local:


replication config to exclude `channelquality_db`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before proceeding with the extraction you should exclude `channelquality_db`
from the replication in addition to the already excluded `tmp_ligs_offline_db`.


Check free space on the server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before extraction check the free space on the server.
The uncompressed DB directory is 9.2G and tarball 2.3G
so you should have at least 25 G free in order to proceed.


Get the tarball and check digest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grab the tarball from S, and check its digest matches those above::

    dybdb1> mkdir -p /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/
    dybdb1> scp S:/data/var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz
             ## its 2.3 GB, so will probably take 30-60 min 

    dybdb1> md5sum  /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz    ## should be 2631bcc9b9c747e238338a4b50c04ad5


Remove empty `channelquality_db`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drop the empty "channelquality_db" database::

    mysql > status                               # check are talking to dybdb1
    mysql > drop database channelquality_db ;
    mysql > select @@datadir ;                  # check where mysql keeps its data, I expect /data/mysql 


Perform extraction into mysql datadir
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From that datadir, check the paths within the tarball, and then extract it should create directory "channelquality_db"::

    dybdb1 > cd /data/mysql
    dybdb1 > tar ztvf  /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz   # check the paths of what will be extracted
              ## a minute or so

    dybdb1 > tar zxvf  /var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz   # do the extraction, creating channelquality_db  
              ## took less than 5 min on belle7 




Basic checks of `channelquality_db` on belle7
------------------------------------------------

The RUNNO/FILENO duplication is from different site/subsite presumably::

    mysql> show tables ;
    +-----------------------------+
    | Tables_in_channelquality_db |
    +-----------------------------+
    | DqChannel                   | 
    | DqChannelStatus             | 
    | DqChannelStatusVld          | 
    | DqChannelVld                | 
    | LOCALSEQNO                  | 
    +-----------------------------+
    5 rows in set (0.00 sec)

    mysql> 
    mysql> 
    mysql> select SEQNO,count(*) N,RUNNO,FILENO from DqChannelStatus group by SEQNO order by SEQNO desc limit 10 ;
    +--------+-----+-------+--------+
    | SEQNO  | N   | RUNNO | FILENO |
    +--------+-----+-------+--------+
    | 323573 | 192 | 38860 |    284 | 
    | 323572 | 192 | 38886 |    343 | 
    | 323571 | 192 | 38886 |    343 | 
    | 323570 | 192 | 38860 |    285 | 
    | 323569 | 192 | 38860 |    285 | 
    | 323568 | 192 | 38886 |    340 | 
    | 323567 | 192 | 38886 |    340 | 
    | 323566 | 192 | 38886 |    336 | 
    | 323565 | 192 | 38886 |    336 | 
    | 323564 | 192 | 38886 |    339 | 
    +--------+-----+-------+--------+
    10 rows in set (0.02 sec)

    mysql> select SEQNO,count(*) N,RUNNO,FILENO from DqChannel group by SEQNO order by SEQNO desc limit 10 ;
    +--------+-----+-------+--------+
    | SEQNO  | N   | RUNNO | FILENO |
    +--------+-----+-------+--------+
    | 323573 | 192 | 38860 |    284 | 
    | 323572 | 192 | 38886 |    343 | 
    | 323571 | 192 | 38886 |    343 | 
    | 323570 | 192 | 38860 |    285 | 
    | 323569 | 192 | 38860 |    285 | 
    | 323568 | 192 | 38886 |    340 | 
    | 323567 | 192 | 38886 |    340 | 
    | 323566 | 192 | 38886 |    336 | 
    | 323565 | 192 | 38886 |    336 | 
    | 323564 | 192 | 38886 |    339 | 
    +--------+-----+-------+--------+
    10 rows in set (0.05 sec)


Large Table Comparisons using grouped digests
------------------------------------------------

* http://stackoverflow.com/questions/3102972/mysql-detecting-changes-in-data-with-a-hash-function-over-a-part-of-table

::

    mysql> select SEQNO,ROW_COUNTER,md5(concat_ws(",",RUNNO,FILENO,CHANNELID,OCCUPANCY,DADCMEAN,DADCRMS,HVMEAN,HVRMS)) from DqChannel limit 10 ;
    +-------+-------------+------------------------------------------------------------------------------------+
    | SEQNO | ROW_COUNTER | md5(concat_ws(",",RUNNO,FILENO,CHANNELID,OCCUPANCY,DADCMEAN,DADCRMS,HVMEAN,HVRMS)) |
    +-------+-------------+------------------------------------------------------------------------------------+
    |     1 |           1 | 6f05eeae022b72a59ce109702579e963                                                   | 
    |     1 |           2 | 237af55b149a4113cb76ba211e9c780c                                                   | 
    |     1 |           3 | 2c0f04d886247c1b2332d5de4343f121                                                   | 
    |     1 |           4 | f25c8f6f0f3863b39549c5b70f4d0d7b                                                   | 
    |     1 |           5 | 38da30183a502faa5edffa0b66a9d7fd                                                   | 
    |     1 |           6 | a33ee7df209680d0bc5c24587d0b7b69                                                   | 
    |     1 |           7 | 606c4aa7971b46ab8535d0f1c436100c                                                   | 
    |     1 |           8 | 614c10c35498c3181d560a193e210a55                                                   | 
    |     1 |           9 | 70b9b4b77d1cea95410a4a72628b6114                                                   | 
    |     1 |          10 | 3aac210ae7274d3e34e6f0b55af66f58                                                   | 
    +-------+-------------+------------------------------------------------------------------------------------+
    10 rows in set (0.00 sec)


::

    mysql> select 33*192 ;
    +--------+
    | 33*192 |
    +--------+
    |   6336 | 
    +--------+
    1 row in set (0.00 sec)

    mysql> show variables like '%group%' ;
    +----------------------------+-------+
    | Variable_name              | Value |
    +----------------------------+-------+
    | group_concat_max_len       | 1024  | 
    | innodb_log_files_in_group  | 2     | 
    | innodb_log_group_home_dir  | ./    | 
    | innodb_mirrored_log_groups | 1     | 
    +----------------------------+-------+
    4 rows in set (0.00 sec)

    mysql> set @@group_concat_max_len = ( 1024 << 3 ) ;
    Query OK, 0 rows affected (0.00 sec)

    mysql> select @@group_concat_max_len ;
    +------------------------+
    | @@group_concat_max_len |
    +------------------------+
    |                   8192 | 
    +------------------------+
    1 row in set (0.00 sec)


A per-SEQNO digest that can be compared between tables in separate DBs that should be the same::

    mysql> select SEQNO,md5(group_concat(md5(concat_ws(",",ROW_COUNTER,RUNNO,FILENO,CHANNELID,OCCUPANCY,DADCMEAN,DADCRMS,HVMEAN,HVRMS)) separator ",")) as digest from DqChannel group by SEQNO limit 10 ;
    +-------+----------------------------------+
    | SEQNO | digest                           |
    +-------+----------------------------------+
    |     1 | 234d05bac921e752a830b725b8e7025d | 
    |     2 | 7c47447ac58bf99cfb1e1619d1ae497b | 
    |     3 | 74dd38ac411bb10f61288f871d9c9bf1 | 
    |     4 | 0ed802219ade8d9aa3b3033d75b2f62f | 
    |     5 | 32a76390c03c6a4bd3d0d1a958725238 | 
    |     6 | 06a6d73226d390e1556450edd8fd54ec | 
    |     7 | ebcdbfb042bf60e934de2d5ee0ec84db | 
    |     8 | d96107391a788a147e861e9c154d9258 | 
    |     9 | 57dac2ede0fe9e48be87896480fd6d84 | 
    |    10 | 29e1fec1affc0ebf814af1777c455078 | 
    +-------+----------------------------------+
    10 rows in set (0.03 sec)

Could also make a full table digest.



Insitu repair and trim
------------------------

DBI trim::

    delete P,V from CableMap P join CableMapVld V on P.SEQNO = V.SEQNO  where P.SEQNO > 398 ;



Aside on MySQL comments
------------------------

Comments in the mysql log can be handy for seeing what commands lead to what SQL in the log::

    [blyth@belle7 ~]$ echo "/* hello without -c does not get into the mysql log */" | mysql  
    [blyth@belle7 ~]$ echo "/* hello with -c gets into the mysql log */" | mysql -c 

How to do that from *mysql-python*::

    In [2]: from DybPython import DB
    In [3]: db = DB()
    In [4]: db("/* hello from DybPython.DB */")
    Out[4]: ()

OR what about DBI::

    In [5]: from DybDbi import gDbi
    In [7]: gDbi.comment("/* hello from gDbi.comment */")    ## HMM NOT WORKING ANY MORE ?



Review belle7 Databases
--------------------------

database summaries
~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 DybPython]$ ./dbsrv.py channelquality_db summary

    channelquality_db
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TABLE_NAME                      TABLE_ROWS  CREATE_TIME                     CHECK_TIME                    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DqChannel                       62126016    2013-05-30 18:52:51             2013-05-30 18:52:51           
    DqChannelStatus                 62126016    2013-05-30 18:17:42             2013-05-30 18:17:42           
    DqChannelStatusVld              323573      2013-05-30 18:52:44             None                          
    DqChannelVld                    323573      2013-05-30 19:34:55             None                          
    LOCALSEQNO                      3           2013-05-30 19:35:02             None                          
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    [blyth@belle7 DybPython]$ ./dbsrv.py tmp_ligs_offline_db_\\d summary      ## regexp argument to dbsrv.py

    tmp_ligs_offline_db_0
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TABLE_NAME                      TABLE_ROWS  CREATE_TIME                     CHECK_TIME                    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DqChannel                       65489088    2013-05-27 13:10:54             2013-05-27 13:26:26           
    DqChannelStatus                 65436731    2013-05-27 13:36:35             2013-05-27 13:51:36           
    DqChannelStatusVld              341125      2013-02-04 16:07:56             2013-05-13 13:16:02           
    DqChannelVld                    341089      2013-02-04 16:07:51             None                          
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tmp_ligs_offline_db_1    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TABLE_NAME                      TABLE_ROWS  CREATE_TIME                     CHECK_TIME                    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DqChannelStatus                 65436672    2013-05-23 14:03:34             None                          
    DqChannelStatusVld              341124      2013-05-23 14:13:29             None                          
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tmp_ligs_offline_db_2
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TABLE_NAME                      TABLE_ROWS  CREATE_TIME                     CHECK_TIME                    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DqChannel                       62126016    2013-05-29 14:59:36             2013-05-29 14:59:37           
    DqChannelStatus                 62126016    2013-05-29 14:26:08             2013-05-29 14:26:09           
    DqChannelStatusVld              323573      2013-05-29 14:59:30             None                          
    DqChannelVld                    323573      2013-05-29 15:38:21             None                          
    LOCALSEQNO                      3           2013-05-29 15:38:27             None                          
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tmp_ligs_offline_db_3
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TABLE_NAME                      TABLE_ROWS  CREATE_TIME                     CHECK_TIME                    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DqChannel                       62126016    2013-05-29 18:15:48             2013-05-29 18:15:49           
    DqChannelStatus                 62126016    2013-05-29 17:42:09             2013-05-29 17:42:10           
    DqChannelStatusVld              323573      2013-05-29 18:15:42             None                          
    DqChannelVld                    323573      2013-05-29 18:54:28             None                          
    LOCALSEQNO                      3           2013-05-29 18:54:34             None                          
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tmp_ligs_offline_db_4
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TABLE_NAME                      TABLE_ROWS  CREATE_TIME                     CHECK_TIME                    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DqChannel                       62126016    2013-05-30 13:54:33             2013-05-30 14:11:53           
    DqChannelStatus                 62126016    2013-05-30 14:11:54             2013-05-30 14:26:55           
    DqChannelStatusVld              323573      2013-05-30 14:26:56             None                          
    DqChannelVld                    323573      2013-05-30 14:26:58             None                          
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tmp_ligs_offline_db_5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TABLE_NAME                      TABLE_ROWS  CREATE_TIME                     CHECK_TIME                    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DqChannel                       9600        2013-06-03 19:44:16             2013-06-03 19:44:16           
    DqChannelStatus                 9600        2013-06-03 19:44:17             2013-06-03 19:44:17           
    DqChannelStatusVld              50          2013-06-03 19:44:17             None                          
    DqChannelVld                    50          2013-06-03 19:44:17             None                          
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [blyth@belle7 DybPython]$ 



`tmp_ligs_offline_db_0`
~~~~~~~~~~~~~~~~~~~~~~~~

#. DB in which `DqChannelStatus` was repaired, and `DqChannel` added in, both from hotcopy tarballs
#. retains  `AUTO_INCREMENT=341126`  

`tmp_ligs_offline_db_1`
~~~~~~~~~~~~~~~~~~~~~~~~

#. freshly created DB populated via the mysqldump obtained from `_0` with the bad SEQNO excluded 
#. `AUTO_INCREMENT` not preserved

`tmp_ligs_offline_db_2`
~~~~~~~~~~~~~~~~~~~~~~~~

#. four tables with faked LOCALSEQNO, obtained from a mysqldump of `_0` with  `SEQNO <= 323573` 
#. `AUTO_INCREMENT` not preserved

`tmp_ligs_offline_db_3`
~~~~~~~~~~~~~~~~~~~~~~~~

#. looks to be the same as `_2`

`tmp_ligs_offline_db_4`
~~~~~~~~~~~~~~~~~~~~~~~~

#. created while testing dumplocal/loadlocal, omitted LOCALSEQNO
#. `AUTO_INCREMENT=341126`    dbsrv.py dumplocal/loadlocal from `_0` 

::
 
    [blyth@belle7 DybPython]$ time ./dbsrv.py tmp_ligs_offline_db_0 dumplocal ~/tmp_ligs_offline_db_0 --where 'SEQNO <= 323573' -l debug 
    [blyth@belle7 DybPython]$ time ./dbsrv.py tmp_ligs_offline_db_4 loadlocal ~/tmp_ligs_offline_db_0  -l debug --DB_DROP_CREATE -C


`tmp_ligs_offline_db_5`
~~~~~~~~~~~~~~~~~~~~~~~~~

#. `AUTO_INCREMENT=341126`   small partitioned load check with dbsrv.py dumplocal/loadlocal from `_0`

`channelquality_db`
~~~~~~~~~~~~~~~~~~~~~

#. DB created from extraction of belle1 hotcopy tarball `/data/var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_2029.tar.gz`
#. no `AUTO_INCREMENT` as lost that via the mysqldump that created 

`channelquality_db_0`
~~~~~~~~~~~~~~~~~~~~~~~

#. DB created from partitioned dump of `channelquality_db`

Partitioned dumplocal/loadlocal::

    #17 19 * * * ( $DYBPYTHON_DIR/dbsrv.py channelquality_db dumplocal /tmp/cq/channelquality_db --partition --partitioncfg 10000,0,33 ) >  $CRONLOG_DIR/dbsrv_dump_.log 2>&1
    #03 20 * * * ( $DYBPYTHON_DIR/dbsrv.py channelquality_db_0 loadlocal  /tmp/cq/channelquality_db --partition --partitioncfg 10000,0,33 --DB_DROP_CREATE -C ) > $CRONLOG_DIR/dbsrv_load_.log 2>&1



testing DB diffing
~~~~~~~~~~~~~~~~~~~~

Dumping 10k SEQNO takes about 30s::


    [blyth@belle7 ~]$ dbsrv tmp_ligs_offline_db_0 dumplocal /tmp/cq/tmp_ligs_offline_db_0 --partition --partitioncfg 10000,0,2 
    2013-06-05 17:50:34,911 __main__ INFO     /* 10000-partition 1  /2  */ SEQNO >= 1 and SEQNO <= 10000 
    2013-06-05 17:50:48,550 __main__ INFO     dumplocal partition 0 SEQNO,0 1:10000 --partitioncfg 10000,0,2 
    2013-06-05 17:50:48,550 __main__ INFO     partition_dumplocal___ /* 10000-partition 1  /2  */ SEQNO >= 1 and SEQNO <= 10000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/0/DqChannel.csv 
    2013-06-05 17:51:05,411 __main__ INFO     partition_dumplocal___ /* 10000-partition 1  /2  */ SEQNO >= 1 and SEQNO <= 10000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/0/DqChannel.csv took 16.86 seconds 
    2013-06-05 17:51:05,411 __main__ INFO     partition_dumplocal___ /* 10000-partition 1  /2  */ SEQNO >= 1 and SEQNO <= 10000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/0/DqChannelStatus.csv 
    2013-06-05 17:51:12,441 __main__ INFO     partition_dumplocal___ /* 10000-partition 1  /2  */ SEQNO >= 1 and SEQNO <= 10000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/0/DqChannelStatus.csv took  7.03 seconds 
    2013-06-05 17:51:12,442 __main__ INFO     partition_dumplocal___ /* 10000-partition 1  /2  */ SEQNO >= 1 and SEQNO <= 10000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/0/DqChannelStatusVld.csv 
    2013-06-05 17:51:12,514 __main__ INFO     partition_dumplocal___ /* 10000-partition 1  /2  */ SEQNO >= 1 and SEQNO <= 10000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/0/DqChannelStatusVld.csv took  0.07 seconds 
    2013-06-05 17:51:12,515 __main__ INFO     partition_dumplocal___ /* 10000-partition 1  /2  */ SEQNO >= 1 and SEQNO <= 10000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/0/DqChannelVld.csv 
    2013-06-05 17:51:12,794 __main__ INFO     partition_dumplocal___ /* 10000-partition 1  /2  */ SEQNO >= 1 and SEQNO <= 10000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/0/DqChannelVld.csv took  0.28 seconds 
    2013-06-05 17:51:13,092 __main__ INFO     /* 10000-partition 2  /2  */ SEQNO >= 10001 and SEQNO <= 20000 
    2013-06-05 17:51:29,136 __main__ INFO     dumplocal partition 1 SEQNO,0 10001:20000 --partitioncfg 10000,0,2 
    2013-06-05 17:51:29,195 __main__ INFO     partition_dumplocal___ /* 10000-partition 2  /2  */ SEQNO >= 10001 and SEQNO <= 20000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/1/DqChannel.csv 
    2013-06-05 17:51:46,344 __main__ INFO     partition_dumplocal___ /* 10000-partition 2  /2  */ SEQNO >= 10001 and SEQNO <= 20000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/1/DqChannel.csv took 17.15 seconds 
    2013-06-05 17:51:46,345 __main__ INFO     partition_dumplocal___ /* 10000-partition 2  /2  */ SEQNO >= 10001 and SEQNO <= 20000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/1/DqChannelStatus.csv 
    2013-06-05 17:51:53,409 __main__ INFO     partition_dumplocal___ /* 10000-partition 2  /2  */ SEQNO >= 10001 and SEQNO <= 20000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/1/DqChannelStatus.csv took  7.06 seconds 
    2013-06-05 17:51:53,410 __main__ INFO     partition_dumplocal___ /* 10000-partition 2  /2  */ SEQNO >= 10001 and SEQNO <= 20000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/1/DqChannelStatusVld.csv 
    2013-06-05 17:51:53,468 __main__ INFO     partition_dumplocal___ /* 10000-partition 2  /2  */ SEQNO >= 10001 and SEQNO <= 20000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/1/DqChannelStatusVld.csv took  0.06 seconds 
    2013-06-05 17:51:53,468 __main__ INFO     partition_dumplocal___ /* 10000-partition 2  /2  */ SEQNO >= 10001 and SEQNO <= 20000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/1/DqChannelVld.csv 
    2013-06-05 17:51:53,674 __main__ INFO     partition_dumplocal___ /* 10000-partition 2  /2  */ SEQNO >= 10001 and SEQNO <= 20000  writing /tmp/cq/tmp_ligs_offline_db_0/10000/1/DqChannelVld.csv took  0.21 seconds 
    [blyth@belle7 ~]$ 


Hmm difference with validity table SEQNO autoincrement ?::

    [blyth@belle7 ~]$ diff -r --brief /tmp/cq/channelquality_db/10000/0/ /tmp/cq/tmp_ligs_offline_db_0/10000/0/
    Files /tmp/cq/channelquality_db/10000/0/DqChannelStatusVld.schema and /tmp/cq/tmp_ligs_offline_db_0/10000/0/DqChannelStatusVld.schema differ
    Files /tmp/cq/channelquality_db/10000/0/DqChannelVld.schema and /tmp/cq/tmp_ligs_offline_db_0/10000/0/DqChannelVld.schema differ
    [blyth@belle7 ~]$ 
    [blyth@belle7 ~]$ diff /tmp/cq/channelquality_db/10000/0/DqChannelStatusVld.schema   /tmp/cq/tmp_ligs_offline_db_0/10000/0/DqChannelStatusVld.schema -y
    CREATE TABLE `DqChannelStatusVld` (                             CREATE TABLE `DqChannelStatusVld` (
      `SEQNO` int(11) NOT NULL,                                   |   `SEQNO` int(11) NOT NULL auto_increment,
      ... 
      `TIMEEND` datetime NOT NULL,                                    `TIMEEND` datetime NOT NULL,
      PRIMARY KEY  (`SEQNO`)                                          PRIMARY KEY  (`SEQNO`)
    ) ENGINE=MyISAM DEFAULT CHARSET=latin1                        | ) ENGINE=MyISAM AUTO_INCREMENT=341126 DEFAULT CHARSET=latin1[blyth@belle7 ~]$ 


Cause of the diff::

    [blyth@belle7 ~]$ echo show create table DqChannelStatusVld | tmp_ligs_offline_db_0 | grep ENGINE
    ) ENGINE=MyISAM AUTO_INCREMENT=341126 DEFAULT CHARSET=latin1 | 

    [blyth@belle7 ~]$ echo show create table DqChannelStatusVld | channelquality_db  | grep ENGINE
    ) ENGINE=MyISAM DEFAULT CHARSET=latin1 | 

    [blyth@belle1 ~]$ echo show create table DqChannelStatusVld | channelquality_db | grep ENGINE
    ) ENGINE=MyISAM DEFAULT CHARSET=latin1 | 


Checking for autoincrement::

     echo show create table DqChannelStatusVld | tmp_ligs_offline_db_0   # AUTO_INCREMENT=341126   from hotcopy extractions
     echo show create table DqChannelStatusVld | tmp_ligs_offline_db_1   # nope                   
     echo show create table DqChannelStatusVld | tmp_ligs_offline_db_2   # nope
     echo show create table DqChannelStatusVld | tmp_ligs_offline_db_3   # nope
     echo show create table DqChannelStatusVld | tmp_ligs_offline_db_4   # AUTO_INCREMENT=341126    dbsrv.py dumplocal from _0 then loadlocal 
     echo show create table DqChannelStatusVld | tmp_ligs_offline_db_5   # AUTO_INCREMENT=341126    small partitioned check using dbsrv.py dumplocal locallocal
     echo show create table DqChannelStatusVld | channelquality_db       # nope  from dbdumpload.py   
     echo show create table DqChannelStatusVld | channelquality_db_0     # nope


dbdumpload.py is omitting auto increment settings ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 ~]$ dbdumpload.py --no-data --tables "DqChannelVld DqChannelStatusVld" tmp_ligs_offline_db_0 dump /dev/stdout  | sh 




mysql bug wrt dumping `AUTO_INCREMENT`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Somehow `AUTO_INCREMENT` setting got lost OR was included when it should not be

* http://bugs.mysql.com/bug.php?id=20786
* http://bugs.mysql.com/bug.php?id=22941

* http://dev.mysql.com/doc/refman/5.0/en/mysqldump.html


create table and autoincrement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* http://dev.mysql.com/doc/refman/5.0/en/create-table.html

An integer or floating-point column can have the additional attribute
`AUTO_INCREMENT`. When you insert a value of NULL (recommended) or 0 into an
indexed `AUTO_INCREMENT` column, the column is set to the next sequence value.
Typically this is value+1, where value is the largest value for the column
currently in the table. `AUTO_INCREMENT` sequences begin with 1.



The initial `AUTO_INCREMENT` value for the table. In MySQL 5.0, this works for
MyISAM and MEMORY tables. It is also supported for InnoDB as of MySQL 5.0.3. To
set the first auto-increment value for engines that do not support the
`AUTO_INCREMENT` table option, insert a *dummy* row with a value one less than
the desired value after creating the table, and then delete the dummy row.

For engines that support the `AUTO_INCREMENT` table option in `CREATE TABLE`
statements, you can also use `ALTER TABLE tbl_name AUTO_INCREMENT = N` to reset
the `AUTO_INCREMENT` value. The value cannot be set lower than the maximum value
currently in the column.



altering `auto_increment`
^^^^^^^^^^^^^^^^^^^^^^^^^^

* DBI will probably ignore the `AUTO_INCREMENT` and come up with its own next SEQNO
* but can alter table to set it anyhow

::

    mysql> show create table CableMapVld ;
    +-------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Table       | Create Table                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
    +-------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | CableMapVld | CREATE TABLE `CableMapVld` (
      `SEQNO` int(11) NOT NULL auto_increment,
      `TIMESTART` datetime NOT NULL,
      `TIMEEND` datetime NOT NULL,
      `SITEMASK` tinyint(4) default NULL,
      `SIMMASK` tinyint(4) default NULL,
      `SUBSITE` int(11) default NULL,
      `TASK` int(11) default NULL,
      `AGGREGATENO` int(11) default NULL,
      `VERSIONDATE` datetime NOT NULL,
      `INSERTDATE` datetime NOT NULL,
      PRIMARY KEY  (`SEQNO`)
    ) ENGINE=MyISAM AUTO_INCREMENT=476 DEFAULT CHARSET=latin1 | 
    +-------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    1 row in set (0.00 sec)

    mysql> alter table CableMapVld AUTO_INCREMENT = 500 ;
    Query OK, 398 rows affected (0.02 sec)
    Records: 398  Duplicates: 0  Warnings: 0

    mysql> show create table CableMapVld ;
    ...
    ) ENGINE=MyISAM AUTO_INCREMENT=500 DEFAULT CHARSET=latin1 | 

    mysql> alter table CableMapVld AUTO_INCREMENT = 476 ;
    Query OK, 398 rows affected (0.00 sec)
    Records: 398  Duplicates: 0  Warnings: 0

    mysql> show create table CableMapVld ;
    ...
    ) ENGINE=MyISAM AUTO_INCREMENT=476 DEFAULT CHARSET=latin1 | 


::

    mysql> select concat(table_schema,'.',table_name) as dbtable,  auto_increment, create_time, update_time from information_schema.tables where table_schema like 'tmp_ligs_offline_db%' or table_schema like 'channelquality_db%'  ;
    +------------------------------------------+----------------+---------------------+---------------------+
    | dbtable                                  | auto_increment | create_time         | update_time         |
    +------------------------------------------+----------------+---------------------+---------------------+
    | channelquality_db.DqChannel              |           NULL | 2013-05-30 18:52:51 | 2013-05-30 19:34:55 | 
    | channelquality_db.DqChannelStatus        |           NULL | 2013-05-30 18:17:42 | 2013-05-30 18:52:44 | 
    | channelquality_db.DqChannelStatusVld     |           NULL | 2013-05-30 18:52:44 | 2013-05-30 18:52:51 | 
    | channelquality_db.DqChannelVld           |           NULL | 2013-05-30 19:34:55 | 2013-05-30 19:35:02 | 
    | channelquality_db.LOCALSEQNO             |           NULL | 2013-05-30 19:35:02 | 2013-05-30 19:35:02 | 
    | channelquality_db_0.DqChannel            |           NULL | 2013-06-04 20:03:01 | 2013-06-04 21:18:00 | 
    | channelquality_db_0.DqChannelStatus      |           NULL | 2013-06-04 20:03:22 | 2013-06-04 21:19:18 | 
    | channelquality_db_0.DqChannelStatusVld   |           NULL | 2013-06-04 20:03:41 | 2013-06-04 21:19:18 | 
    | channelquality_db_0.DqChannelVld         |           NULL | 2013-06-04 20:03:41 | 2013-06-04 21:19:18 | 
    | tmp_ligs_offline_db_0.DqChannel          |           NULL | 2013-05-27 13:10:54 | 2013-05-27 13:18:17 | 
    | tmp_ligs_offline_db_0.DqChannelStatus    |           NULL | 2013-05-27 13:36:35 | 2013-05-27 13:43:50 | 
    | tmp_ligs_offline_db_0.DqChannelStatusVld |         341126 | 2013-02-04 16:07:56 | 2013-05-20 06:26:55 | 
    | tmp_ligs_offline_db_0.DqChannelVld       |         341090 | 2013-02-04 16:07:51 | 2013-05-20 06:26:54 | 
    | tmp_ligs_offline_db_1.DqChannelStatus    |           NULL | 2013-05-23 14:03:34 | 2013-05-23 14:13:28 | 
    | tmp_ligs_offline_db_1.DqChannelStatusVld |           NULL | 2013-05-23 14:13:29 | 2013-05-23 14:13:36 | 
    | tmp_ligs_offline_db_2.DqChannel          |           NULL | 2013-05-29 14:59:36 | 2013-05-29 15:38:21 | 
    | tmp_ligs_offline_db_2.DqChannelStatus    |           NULL | 2013-05-29 14:26:08 | 2013-05-29 14:59:30 | 
    | tmp_ligs_offline_db_2.DqChannelStatusVld |           NULL | 2013-05-29 14:59:30 | 2013-05-29 14:59:35 | 
    | tmp_ligs_offline_db_2.DqChannelVld       |           NULL | 2013-05-29 15:38:21 | 2013-05-29 15:38:26 | 
    | tmp_ligs_offline_db_2.LOCALSEQNO         |           NULL | 2013-05-29 15:38:27 | 2013-05-29 15:38:27 | 
    | tmp_ligs_offline_db_3.DqChannel          |           NULL | 2013-05-29 18:15:48 | 2013-05-29 18:54:27 | 
    | tmp_ligs_offline_db_3.DqChannelStatus    |           NULL | 2013-05-29 17:42:09 | 2013-05-29 18:15:41 | 
    | tmp_ligs_offline_db_3.DqChannelStatusVld |           NULL | 2013-05-29 18:15:42 | 2013-05-29 18:15:46 | 
    | tmp_ligs_offline_db_3.DqChannelVld       |           NULL | 2013-05-29 18:54:28 | 2013-05-29 18:54:33 | 
    | tmp_ligs_offline_db_3.LOCALSEQNO         |           NULL | 2013-05-29 18:54:34 | 2013-05-29 18:54:34 | 
    | tmp_ligs_offline_db_4.DqChannel          |           NULL | 2013-05-30 13:54:33 | 2013-05-30 14:03:58 | 
    | tmp_ligs_offline_db_4.DqChannelStatus    |           NULL | 2013-05-30 14:11:54 | 2013-05-30 14:19:39 | 
    | tmp_ligs_offline_db_4.DqChannelStatusVld |         341126 | 2013-05-30 14:26:56 | 2013-05-30 14:26:58 | 
    | tmp_ligs_offline_db_4.DqChannelVld       |         341090 | 2013-05-30 14:26:58 | 2013-05-30 14:27:01 | 
    | tmp_ligs_offline_db_5.DqChannel          |           NULL | 2013-06-03 19:44:16 | 2013-06-03 19:44:17 | 
    | tmp_ligs_offline_db_5.DqChannelStatus    |           NULL | 2013-06-03 19:44:17 | 2013-06-03 19:44:17 | 
    | tmp_ligs_offline_db_5.DqChannelStatusVld |         341126 | 2013-06-03 19:44:17 | 2013-06-03 19:44:17 | 
    | tmp_ligs_offline_db_5.DqChannelVld       |         341090 | 2013-06-03 19:44:17 | 2013-06-03 19:44:17 | 
    +------------------------------------------+----------------+---------------------+---------------------+
    33 rows in set (0.01 sec)


    mysql> select max(SEQNO) from tmp_ligs_offline_db_0.DqChannelStatusVld ;
    +------------+
    | max(SEQNO) |
    +------------+
    |     341125 | 
    +------------+
    1 row in set (0.03 sec)

    mysql> select max(SEQNO) from tmp_ligs_offline_db_0.DqChannelVld  ;
    +------------+
    | max(SEQNO) |
    +------------+
    |     341089 | 
    +------------+
    1 row in set (0.04 sec)

    mysql> select concat(table_schema,'.',table_name) as dbtable,  auto_increment, create_time, update_time from information_schema.tables where auto_increment > 10 and auto_increment < 1000 ;
    +--------------------------------------+----------------+---------------------+---------------------+
    | dbtable                              | auto_increment | create_time         | update_time         |
    +--------------------------------------+----------------+---------------------+---------------------+
    | fake_dcs.DBNS_AD_HV_Imon             |            295 | 2012-08-31 09:09:25 | 2012-08-31 09:09:25 | 
    | fake_dcs.DBNS_AD_HV_Pw               |            296 | 2012-08-31 09:09:25 | 2012-08-31 09:09:25 | 
    | fake_dcs.DBNS_AD_HV_Vmon             |            295 | 2012-08-31 09:09:25 | 2012-08-31 09:09:25 | 
    | fake_dcs.ins_location                |             23 | 2012-08-31 09:09:26 | 2012-08-31 09:09:26 | 
    | fix_offline_db.CableMapVld           |            476 | 2013-06-05 19:54:21 | 2013-06-05 19:54:21 | 
    | mydb.auth_permission                 |             34 | 2009-05-27 15:26:38 | 2009-08-14 18:01:09 | 
    | mydb.auth_user_user_permissions      |             27 | 2009-05-27 15:25:54 | 2009-06-08 16:13:34 | 
    | mydb.django_content_type             |             14 | 2009-05-27 15:25:54 | 2009-08-14 18:01:09 | 
    | tmp_cascade_1.DbiDemoData1Vld        |            204 | 2011-08-19 19:03:49 | 2011-08-19 19:03:49 | 
    | tmp_cascade_1.DbiDemoData2Vld        |            635 | 2011-08-19 19:03:49 | 2011-08-19 19:03:49 | 
    | tmp_dbitest_1.DbiDemoData1Vld        |            204 | 2013-06-04 13:05:14 | 2013-06-04 13:05:14 | 
    | tmp_dbitest_1.DbiDemoData2Vld        |            635 | 2013-06-04 13:05:14 | 2013-06-04 13:05:14 | 
    | tmp_dybdbitest_1.DbiDemoData1Vld     |            204 | 2013-06-04 13:05:33 | 2013-06-04 13:05:33 | 
    | tmp_dybdbitest_1.DbiDemoData2Vld     |            635 | 2013-06-04 13:05:33 | 2013-06-04 13:05:33 | 
    | tmp_dybdbitest_2.DbiDemoData1Vld     |            204 | 2013-06-04 13:05:33 | 2013-06-04 13:05:33 | 
    | tmp_dybdbitest_2.DbiDemoData2Vld     |            635 | 2013-06-04 13:05:33 | 2013-06-04 13:05:33 | 
    | tmp_offline_db.DemoVld               |             50 | 2013-05-28 18:26:02 | 2013-05-28 18:39:55 | 
    | tmp_tmp_offline_db_1.DbiDemoData1Vld |            204 | 2011-03-29 18:59:10 | 2011-03-29 18:59:10 | 
    | tmp_tmp_offline_db_1.DbiDemoData2Vld |            635 | 2011-03-29 18:59:10 | 2011-03-29 18:59:10 | 
    +--------------------------------------+----------------+---------------------+---------------------+
    19 rows in set (0.10 sec)

    mysql> 
    mysql> 
    mysql> select max(SEQNO) from tmp_tmp_offline_db_1.DbiDemoData1Vld ;
    +------------+
    | max(SEQNO) |
    +------------+
    |        203 | 
    +------------+
    1 row in set (0.00 sec)

    mysql> select max(SEQNO) from tmp_offline_db.DemoVld  ;
    +------------+
    | max(SEQNO) |
    +------------+
    |         49 | 
    +------------+
    1 row in set (0.00 sec)


altering auto increment
~~~~~~~~~~~~~~~~~~~~~~~~~~

* seems cannot combine the query and alter, even trying to use variables failed

::

    mysql> select max(SEQNO)+1 from CableMapVld ;
    +--------------+
    | max(SEQNO)+1 |
    +--------------+
    |          399 | 
    +--------------+
    1 row in set (0.00 sec)

    mysql> alter table CableMapVld auto_increment = 399 ;
    Query OK, 398 rows affected (0.01 sec)
    Records: 398  Duplicates: 0  Warnings: 0


Test use of partitioned dumplocal for large DB diffing 
-----------------------------------------------------------

Diff all tables partition of 10k SEQNO at a time, taking approx 2-5s.  
Thus 2-3 min to diff 10G of DB.

::

    [blyth@belle7 cq]$  diff -r --brief channelquality_db/10000/_ tmp_ligs_offline_db_0/10000/_                     ## the auto increment issue
    Files channelquality_db/10000/_/DqChannelStatusVld.schema and tmp_ligs_offline_db_0/10000/_/DqChannelStatusVld.schema differ
    Files channelquality_db/10000/_/DqChannelVld.schema and tmp_ligs_offline_db_0/10000/_/DqChannelVld.schema differ

    [blyth@belle7 cq]$ diff -r --brief channelquality_db/10000/1 tmp_ligs_offline_db_0/10000/0                      ## oops must line up the patitions
    Files channelquality_db/10000/1/DqChannel.csv and tmp_ligs_offline_db_0/10000/0/DqChannel.csv differ
    Files channelquality_db/10000/1/DqChannelStatus.csv and tmp_ligs_offline_db_0/10000/0/DqChannelStatus.csv differ
    Files channelquality_db/10000/1/DqChannelStatusVld.csv and tmp_ligs_offline_db_0/10000/0/DqChannelStatusVld.csv differ
    Files channelquality_db/10000/1/DqChannelVld.csv and tmp_ligs_offline_db_0/10000/0/DqChannelVld.csv differ

    [blyth@belle7 cq]$ diff -r --brief channelquality_db/10000/0 tmp_ligs_offline_db_0/10000/0
    [blyth@belle7 cq]$ diff -r --brief channelquality_db/10000/1 tmp_ligs_offline_db_0/10000/1
    [blyth@belle7 cq]$ diff -r --brief channelquality_db/10000/2 tmp_ligs_offline_db_0/10000/2
    [blyth@belle7 cq]$ diff -r --brief channelquality_db/10000/3 tmp_ligs_offline_db_0/10000/3
    [blyth@belle7 cq]$ diff -r --brief channelquality_db/10000/4 tmp_ligs_offline_db_0/10000/4
    [blyth@belle7 cq]$ time diff -r --brief channelquality_db/10000/5 tmp_ligs_offline_db_0/10000/5
    real    0m5.352s
    user    0m0.736s
    sys     0m0.265s

    [blyth@belle7 cq]$ time diff -r --brief channelquality_db/10000/6 tmp_ligs_offline_db_0/10000/6
    real    0m2.658s
    user    0m0.648s
    sys     0m0.235s

    [blyth@belle7 cq]$ time diff -r --brief channelquality_db/10000/7 tmp_ligs_offline_db_0/10000/7
    real    0m6.171s
    user    0m0.686s
    sys     0m0.277s







Validating Recovered DB 
--------------------------

Procedure if had verified backups available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. a verified and daily updated backup would be available on a remote node 
#. on finding corruption in a table, repairs could be performed in situ and 
   the result could be compared against the verified backup.

   * if losses/problems were seen as a result of the repair would need to 
     revert to the backup and refill the entries that were lost 

Our procedure without verified backups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: Poor situation arising due to a non backed-up `tmp_` DB morphing into a critical one  


#. corruption found
#. hotcopy of the tables made and transferred to a remote node, 
   where a repair was done, appearing to be successful

   * but cannot be sure of that without a backup to compare against ? 

#. forced to assume the 


Strategy
~~~~~~~~~~

#. get confidence in the techniques developed by validating 
   the three non-crashed tables up to `SEQNO <= 323573` by comparison of partitioned dumps::

      DqChannel
      DqChannelVld
      DqChannelStatusVld

   success of this indicates are doing no harm, for these three


Insitu repair
~~~~~~~~~~~~~~~~~

Doing a repair on the server and comparing with 
the repair done on belle7 would give a crosscheck,
but not certainty.

Although the repair was simple to do on belle7 (and took only ~4 min)
it can potentially loose all data in the table. 
Thus you should never normally do it without having a verified backup.


Databases compared
~~~~~~~~~~~~~~~~~~~~~

===========  ===========================  ========================
home sect    sect                         note     
===========  ===========================  ========================
dybdb1_ligs   tmp_ligs_offline_db_dybdb1   original on dybdb1
dybdb2_ligs   channelquality_db_dybdb2     recovered on dybdb2
loopback      channelquality_db_belle7     recovered onto belle7 from hotcopy created on belle1
===========  ===========================  ========================


Partitioning
~~~~~~~~~~~~~~~

Partition tables in SEQNO chunks. 
The chunking allows DB diffs to be made using filesystem "diff -r --brief" 
applied to the chunk directories.  
This results in drastically faster diffing than 
for example creating digests from queries.
Also this will allow efficient incremental 
backups of very large tables.

Using 

* :dybsvn:`source:dybgaudi/trunk/DybPython/python/DybPython/dbsrv.py`
* :dybsvn:`source:dybgaudi/trunk/DybPython/python/DybPython/diff.py`

Partitioned dumping (in 10000 SEQNO chunks) created CSV files for each table::

    [blyth@belle7 10000]$ pwd
    /tmp/cq/channelquality_db_dybdb2/10000
    [blyth@belle7 10000]$ l 0/
    total 115512
    -rw-r--r-- 1 blyth blyth   1038894 Jun  6 18:54 DqChannelStatusVld.csv
    -rw-r--r-- 1 blyth blyth   1038894 Jun  6 18:54 DqChannelVld.csv
    -rw-r--r-- 1 blyth blyth 116074010 Jun  6 18:54 DqChannel.csv
    [blyth@belle7 10000]$ 

This allowed comparison using commands like::

     diff -r --brief /tmp/cq/channelquality_db_belle7/10000/0 /tmp/cq/channelquality_db_dybdb2/10000/0

Only an insignificant formatting difference was encountered in one file

* http://belle7.nuu.edu.tw/oum/api/dbsrv/#oops-a-difference-but-its-just-different-formatting-of-0-0001-or-1e-04

More details at 

* http://belle7.nuu.edu.tw/oum/api/dbsrv/       


