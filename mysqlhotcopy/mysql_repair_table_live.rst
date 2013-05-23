
MySQL repair table live
=========================

.. contents:: :local:

Timeline Summary
------------------

April 30
       corruption occurs (assumed to be due to a killed KUP job) it goes unnoticed the table continuing to be written to 
May 13
       while performing a test compression run on DqChannelStatus corrupt SEQNO 323575 in DqChannelStatus is discovered :dybsvn:`ticket:1347#comment:20`   
May 14
       begin development of :env:`source:trunk/mysqlhotcopy/mysqlhotcopy.py` with `hotcopy/archive/extract/transfer` capabilities
May 15
       formulate plan of action the first step of which is making a hotcopy backup 
May 16 
       start working with Qiumei get to `mysqlhotcopy.py` operational on dybdb1.ihep.ac.cn, Miao notifies us that CQ filling is suspended
May 17-23
       development via email (~18 email exchanges and ~20 env commits later, numerous issues every one of which required email exchange and related delays)
May 19
       `2013-05-19 08:22:20` CQ filling resumes (contrary to expectations), but writes are Validity only due to the crashed payload table
May 20
       1st attempt to perform hotcopy on dybdb1 meets error due to crashed table, originally thought that the hotcopy *flush* might have
       caused the crashed state, but the timing of the last validity insert `2013-05-19 22:26:55` is suggestive that the crash was due to this
May 21
       Gaosong notes that cannot access the DqChannelStatus table at all, due to crashed status
May 23
       finally a coldcopy (hotcopy fails due to crashed table) tarball transferred to NUU, and is extracted into DB and repaired 


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








Repair on primary server, or propagate the fixed ?
----------------------------------------------------

