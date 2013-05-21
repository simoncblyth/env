Repair Table
===============



Explore corruption on belle7.

.. contents:: :local:


References
-----------

* :google:`mysql repair corruption`

* http://dev.mysql.com/doc/refman/5.0/en/repair-table.html
* http://dev.mysql.com/doc/refman/5.1/en/repair-table.html


MySQL Versions and `USE_FRM` option
--------------------------------------

::

   dybdb1.ihep.ac.cn        5.0.45-community-log MySQL Community Edition (GPL)
   belle7.nuu.edu.tw        5.0.77-log Source distribution
   cms01.phys.ntu.edu.tw    4.1.22-log


From http://dev.mysql.com/doc/refman/5.0/en/repair-table.html

As of MySQL 5.0.62, if you use `USE_FRM` for a table that was created by a
different version of the MySQL server than the one you are currently running,
REPAIR TABLE will not attempt to repair the table. In this case, the result set
returned by REPAIR TABLE contains a line with a `Msg_type` value of `error` and a
`Msg_text` value of `Failed repairing incompatible .FRM file`.

   * so on belle7 I cannot use `USE_FRM` to repair the table from dybdb1

Prior to MySQL 5.0.62, do not use `USE_FRM` if your table was created by a
different version of the MySQL server. Doing so risks the loss of all rows in
the table. It is particularly dangerous to use `USE_FRM` after the server returns
this message:

   * the tables appear have been created `2013-02-04` so seems no issue with version differences for dybdb1 repair using `USE_FRM` 

::

    mysql> select table_name, table_type, engine, version, table_rows, data_length, max_data_length, index_length, data_free, create_time, update_time, check_time from information_schema.tables where table_schema = 'tmp_ligs_offline_db' ;
    +-----------------------+------------+-----------+---------+------------+-------------+-------------------+--------------+-----------+---------------------+---------------------+---------------------+
    | table_name            | table_type | engine    | version | table_rows | data_length | max_data_length   | index_length | data_free | create_time         | update_time         | check_time          |
    +-----------------------+------------+-----------+---------+------------+-------------+-------------------+--------------+-----------+---------------------+---------------------+---------------------+
    | ChannelQuality        | BASE TABLE | MyISAM    |      10 |    1745856 |    24441984 |  3940649673949183 |     25170944 |         0 | 2013-04-22 12:50:10 | 2013-04-22 23:32:27 | NULL                | 
    | ChannelQualityVld     | BASE TABLE | MyISAM    |      10 |       9093 |      463743 | 14355223812243455 |        96256 |         0 | 2013-04-22 12:50:10 | 2013-04-22 23:32:27 | NULL                | 
    | DaqRawDataFileInfo    | BASE TABLE | FEDERATED |      10 |     310821 |    70867188 |                 0 |            0 |         0 | NULL                | 1970-01-01 08:33:33 | NULL                | 
    | DaqRawDataFileInfoVld | BASE TABLE | FEDERATED |      10 |     310821 |    13986945 |                 0 |            0 |         0 | NULL                | 1970-01-01 08:33:33 | NULL                | 
    | DqChannel             | BASE TABLE | MyISAM    |      10 |   65489088 |  2750541696 | 11821949021847551 |   1015181312 |         0 | 2013-02-04 16:07:51 | 2013-05-20 06:26:54 | NULL                | 
    | DqChannelStatus       | BASE TABLE | NULL      |    NULL |       NULL |        NULL |              NULL |         NULL |      NULL | NULL                | NULL                | NULL                | 
    | DqChannelStatusVld    | BASE TABLE | MyISAM    |      10 |     341125 |    17397375 | 14355223812243455 |      3826688 |         0 | 2013-02-04 16:07:56 | 2013-05-20 06:26:55 | 2013-05-13 13:16:02 | 
    | DqChannelVld          | BASE TABLE | MyISAM    |      10 |     341089 |    17395539 | 14355223812243455 |      3606528 |         0 | 2013-02-04 16:07:51 | 2013-05-20 06:26:54 | NULL                | 
    | LOCALSEQNO            | BASE TABLE | MyISAM    |      10 |          4 |         276 | 19421773393035263 |         2048 |         0 | 2013-02-04 16:09:33 | 2013-05-20 06:26:54 | NULL                | 
    +-----------------------+------------+-----------+---------+------------+-------------+-------------------+--------------+-----------+---------------------+---------------------+---------------------+
    9 rows in set (0.09 sec)



Repairs and replication
------------------------

From http://dev.mysql.com/doc/refman/5.0/en/repair-table.html

By default, the server writes `REPAIR TABLE` statements to the binary log so that
they replicate to replication slaves. To suppress logging, specify the optional
`NO_WRITE_TO_BINLOG` keyword or its alias `LOCAL`.

* this DB is skipped from replication, so presumably no problem BUT should perhaps use `REPAIR LOCAL TABLE DqChannelStatus` 



Create a throwaway DB
-----------------------

::

    mysqlhotcopy.py --regex "^DqChannelPacked"  -l debug --ALLOWEXTRACT --flattop -C --rename tmp_offline_db_ext tmp_offline_db coldcopy archive examine extract  


Verify accessible before being detructive
------------------------------------------

::

    mysql> use  tmp_offline_db_ext  
    Database changed
    mysql> show tables ;
    +------------------------------+
    | Tables_in_tmp_offline_db_ext |
    +------------------------------+
    | DqChannelPacked              | 
    | DqChannelPackedVld           | 
    +------------------------------+
    2 rows in set (0.00 sec)

    mysql> select count(*) from DqChannelPacked ;   
    +----------+
    | count(*) |
    +----------+
    |   323000 | 
    +----------+
    1 row in set (0.00 sec)

    mysql> select count(*) from DqChannelPackedVld ;
    +----------+
    | count(*) |
    +----------+
    |   323000 | 
    +----------+
    1 row in set (0.00 sec)

    mysql> select * from DqChannelPackedVld order by SEQNO desc limit 5 ;
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | SEQNO  | TIMESTART           | TIMEEND             | SITEMASK | SIMMASK | SUBSITE | TASK | AGGREGATENO | VERSIONDATE         | INSERTDATE          |
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    | 323000 | 2013-04-27 23:07:43 | 2013-04-27 23:29:31 |        4 |       1 |       2 |    0 |          -1 | 2013-04-27 23:07:43 | 2013-05-11 12:18:46 | 
    | 322999 | 2013-04-27 23:07:43 | 2013-04-27 23:29:31 |        4 |       1 |       4 |    0 |          -1 | 2013-04-27 23:07:43 | 2013-05-11 12:18:45 | 
    | 322998 | 2013-04-27 23:44:38 | 2013-04-27 23:54:30 |        1 |       1 |       1 |    0 |          -1 | 2013-04-27 23:44:38 | 2013-05-11 12:18:45 | 
    | 322997 | 2013-04-27 23:44:38 | 2013-04-27 23:54:30 |        1 |       1 |       2 |    0 |          -1 | 2013-04-27 23:44:38 | 2013-05-11 12:18:44 | 
    | 322996 | 2013-04-28 00:10:09 | 2013-04-28 00:22:35 |        2 |       1 |       1 |    0 |          -1 | 2013-04-28 00:10:09 | 2013-05-11 12:18:44 | 
    +--------+---------------------+---------------------+----------+---------+---------+------+-------------+---------------------+---------------------+
    5 rows in set (0.00 sec)

    mysql> select * from DqChannelPacked order by SEQNO desc limit 5 ;
    +--------+-------------+-------+--------+------------+------------+------------+------------+------------+------------+-------+
    | SEQNO  | ROW_COUNTER | RUNNO | FILENO | MASK0      | MASK1      | MASK2      | MASK3      | MASK4      | MASK5      | MASK6 |
    +--------+-------------+-------+--------+------------+------------+------------+------------+------------+------------+-------+
    | 323000 |           1 | 38878 |    115 | 2147483647 | 2147483647 | 2147483647 | 2147483647 | 2147483647 | 2147483647 |    63 | 
    | 322999 |           1 | 38878 |    115 | 2147483647 | 2147483647 | 2139095039 | 2147483647 | 2147483647 | 2147483647 |    63 | 
    | 322998 |           1 | 38886 |    229 | 2147483647 | 2147483647 | 2147483647 | 2147483647 | 2147483647 | 2147483647 |    63 | 
    | 322997 |           1 | 38886 |    229 | 2147483647 | 2147483647 | 2147483647 | 2147483647 | 2147483647 | 2147483647 |    63 | 
    | 322996 |           1 | 38860 |    198 | 2147483647 | 2147483647 | 2147483647 | 2147483647 | 2147483647 | 2147483647 |    63 | 
    +--------+-------------+-------+--------+------------+------------+------------+------------+------------+------------+-------+
    5 rows in set (0.00 sec)


Be destructive, delete the MYI index file for a table
------------------------------------------------------

::

    [root@belle7 tmp_offline_db_ext]# pwd
    /var/lib/mysql/tmp_offline_db_ext
    [root@belle7 tmp_offline_db_ext]# ll
    total 38484
    -rw-rw----  1 mysql mysql     8908 May 10 18:18 DqChannelPackedVld.frm
    -rw-rw----  1 mysql mysql     8896 May 10 18:18 DqChannelPacked.frm
    -rw-rw----  1 mysql mysql 16473000 May 11 20:18 DqChannelPackedVld.MYD
    -rw-rw----  1 mysql mysql 14858000 May 11 20:18 DqChannelPacked.MYD
    -rw-rw----  1 mysql mysql  4658176 May 13 13:08 DqChannelPacked.MYI
    -rw-rw----  1 mysql mysql  3314688 May 14 15:04 DqChannelPackedVld.MYI
    drwxr-x---  2 mysql mysql     4096 May 16 17:11 .
    drwxr-xr-x 40 mysql mysql     4096 May 20 19:54 ..
    [root@belle7 tmp_offline_db_ext]# rm DqChannelPacked.MYI
    rm: remove regular file `DqChannelPacked.MYI'? y
    [root@belle7 tmp_offline_db_ext]# 



Repairing the damage
---------------------

* :google:`repair mysql corruption`
* http://www.databasejournal.com/features/mysql/article.php/10897_3300511_2/Repairing-Database-Corruption-in-MySQL.htm

Appears to work OK for a while (memory cache ?) then after flushing::

    mysql> flush tables ;
    Query OK, 0 rows affected (0.02 sec)

    mysql> select count(*) from DqChannelPacked    ;
    ERROR 1017 (HY000): Can't find file: 'DqChannelPacked' (errno: 2)


Check table repeats that error and repair table fails to clear it::

    mysql> check table  DqChannelPacked    ;
    +------------------------------------+-------+----------+-----------------------------------------------+
    | Table                              | Op    | Msg_type | Msg_text                                      |
    +------------------------------------+-------+----------+-----------------------------------------------+
    | tmp_offline_db_ext.DqChannelPacked | check | Error    | Can't find file: 'DqChannelPacked' (errno: 2) | 
    | tmp_offline_db_ext.DqChannelPacked | check | error    | Corrupt                                       | 
    +------------------------------------+-------+----------+-----------------------------------------------+
    2 rows in set (0.00 sec)

    mysql> REPAIR TABLE DqChannelPacked    ;
    +------------------------------------+--------+----------+-----------------------------------------------+
    | Table                              | Op     | Msg_type | Msg_text                                      |
    +------------------------------------+--------+----------+-----------------------------------------------+
    | tmp_offline_db_ext.DqChannelPacked | repair | Error    | Can't find file: 'DqChannelPacked' (errno: 2) | 
    | tmp_offline_db_ext.DqChannelPacked | repair | error    | Corrupt                                       | 
    +------------------------------------+--------+----------+-----------------------------------------------+
    2 rows in set (0.00 sec)

    mysql> 
    mysql> check table  DqChannelPacked    ;
    +------------------------------------+-------+----------+-----------------------------------------------+
    | Table                              | Op    | Msg_type | Msg_text                                      |
    +------------------------------------+-------+----------+-----------------------------------------------+
    | tmp_offline_db_ext.DqChannelPacked | check | Error    | Can't find file: 'DqChannelPacked' (errno: 2) | 
    | tmp_offline_db_ext.DqChannelPacked | check | error    | Corrupt                                       | 
    +------------------------------------+-------+----------+-----------------------------------------------+
    2 rows in set (0.00 sec)


With the `USE_FRM` succeed to repair the table, which recreated the MYI index that I deleted.
Ordinarily `USE_FRM` is not advised unless the other repair techniques fail, see http://dev.mysql.com/doc/refman/5.0/en/repair-table.html
::

    mysql> REPAIR TABLE  DqChannelPacked USE_FRM ;
    +------------------------------------+--------+----------+-----------------------------------------+
    | Table                              | Op     | Msg_type | Msg_text                                |
    +------------------------------------+--------+----------+-----------------------------------------+
    | tmp_offline_db_ext.DqChannelPacked | repair | warning  | Number of rows changed from 0 to 323000 | 
    | tmp_offline_db_ext.DqChannelPacked | repair | status   | OK                                      | 
    +------------------------------------+--------+----------+-----------------------------------------+
    2 rows in set (0.42 sec)

    mysql> check table DqChannelPacked ;
    +------------------------------------+-------+----------+----------+
    | Table                              | Op    | Msg_type | Msg_text |
    +------------------------------------+-------+----------+----------+
    | tmp_offline_db_ext.DqChannelPacked | check | status   | OK       | 
    +------------------------------------+-------+----------+----------+
    1 row in set (0.14 sec)








