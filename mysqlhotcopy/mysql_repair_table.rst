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



MyISAM repairs
------------------

From http://dev.mysql.com/doc/refman/5.0/en/myisam-repair.html

::

    [root@belle7 tmp_offline_db_ext]# man myisamchk
    [root@belle7 tmp_offline_db_ext]# myisamchk -vvv *.MYI
    Warning: option 'key_buffer_size': unsigned value 18446744073709551615 adjusted to 4294963200
    Warning: option 'read_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'write_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'sort_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Checking MyISAM file: DqChannelPacked.MYI
    Data records:  323000   Deleted blocks:       0
    - check file-size
    - check record delete-chain
    No recordlinks
    - check key delete-chain
    block_size 1024:
    - check index reference
    - check data record references index: 1

    ---------

    Checking MyISAM file: DqChannelPackedVld.MYI
    Data records:  323000   Deleted blocks:       0
    - check file-size
    - check record delete-chain
    No recordlinks
    - check key delete-chain
    block_size 1024:
    - check index reference
    - check data record references index: 1
    [root@belle7 tmp_offline_db_ext]# 




Following sections `Stage 3: Difficult repair` and `Stage 2: Easy safe repair`
--------------------------------------------------------------------------------

* http://dev.mysql.com/doc/refman/5.0/en/myisam-repair.html

move MYI and MYD into keep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulate a missing MYI::

    [root@belle7 tmp_offline_db_ext]# mkdir ../tmp_offline_db_ext_keep
    [root@belle7 tmp_offline_db_ext]# mv DqChannelPacked.MYI DqChannelPacked.MYD ../tmp_offline_db_ext_keep
    [root@belle7 tmp_offline_db_ext]# ll  ../tmp_offline_db_ext_keep
    total 19072
    -rw-rw----  1 mysql mysql 14858000 May 21 13:43 DqChannelPacked.MYD
    -rw-rw----  1 mysql mysql  4621312 May 21 13:46 DqChannelPacked.MYI
    drwxr-xr-x 41 mysql mysql     4096 May 22 18:56 ..
    drwxr-xr-x  2 root  root      4096 May 22 18:57 .


truncate the moved table, recreating a 1024 byte MYI and empty MYD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    mysql> show tables ;
    +------------------------------+
    | Tables_in_tmp_offline_db_ext |
    +------------------------------+
    | DqChannelPacked              | 
    | DqChannelPackedVld           | 
    +------------------------------+
    2 rows in set (0.00 sec)

    mysql> SET autocommit=1;
    Query OK, 0 rows affected (0.00 sec)

    mysql> truncate table DqChannelPacked  ;
    Query OK, 0 rows affected (0.02 sec)

    mysql> quit
    Bye

::

    [root@belle7 tmp_offline_db_ext]# ll
    total 19392
    -rw-rw----  1 mysql mysql     8908 May 10 18:18 DqChannelPackedVld.frm
    -rw-rw----  1 mysql mysql     8896 May 10 18:18 DqChannelPacked.frm
    -rw-rw----  1 mysql mysql 16473000 May 11 20:18 DqChannelPackedVld.MYD
    -rw-rw----  1 mysql mysql  3314688 May 14 15:04 DqChannelPackedVld.MYI
    drwxr-xr-x 41 mysql mysql     4096 May 22 18:56 ..
    -rw-rw----  1 mysql mysql     1024 May 22 18:59 DqChannelPacked.MYI
    -rw-rw----  1 mysql mysql        0 May 22 18:59 DqChannelPacked.MYD
    drwxr-x---  2 mysql mysql     4096 May 22 18:59 .


Copy the MYD back from keep ontop of the empty MYD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy the old data file back onto the newly created data file. (Do not just move
the old file back onto the new file. You want to retain a copy in case
something goes wrong.)

::

    [root@belle7 tmp_offline_db_ext]# cp ../tmp_offline_db_ext_keep/DqChannelPacked.MYD .
    cp: overwrite `./DqChannelPacked.MYD'? y

    [root@belle7 tmp_offline_db_ext]# ll
    total 33924
    -rw-rw----  1 mysql mysql     8908 May 10 18:18 DqChannelPackedVld.frm
    -rw-rw----  1 mysql mysql     8896 May 10 18:18 DqChannelPacked.frm
    -rw-rw----  1 mysql mysql 16473000 May 11 20:18 DqChannelPackedVld.MYD
    -rw-rw----  1 mysql mysql  3314688 May 14 15:04 DqChannelPackedVld.MYI
    drwxr-xr-x 41 mysql mysql     4096 May 22 18:56 ..
    -rw-rw----  1 mysql mysql     1024 May 22 18:59 DqChannelPacked.MYI
    drwxr-x---  2 mysql mysql     4096 May 22 18:59 .
    -rw-rw----  1 mysql mysql 14858000 May 22 19:06 DqChannelPacked.MYD

The result so far is a drastically shrunk MYI.  


Repopulate the index with `myisamchk -r -q`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From `Stage 2: Easy safe repair` of http://dev.mysql.com/doc/refman/5.0/en/myisam-repair.html

This attempts to repair the index file without touching the data file. If the
data file contains everything that it should and the delete links point at the
correct locations within the data file, this should work, and the table is
fixed.


::

    [root@belle7 tmp_offline_db_ext]# myisamchk -r -q DqChannelPacked
    Warning: option 'key_buffer_size': unsigned value 18446744073709551615 adjusted to 4294963200
    Warning: option 'read_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'write_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'sort_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    - check record delete-chain
    - recovering (with sort) MyISAM-table 'DqChannelPacked'
    Data records: 0
    - Fixing index 1
    Data records: 323000
    [root@belle7 tmp_offline_db_ext]# 

Those warnings are a know bug http://bugs.mysql.com/bug.php?id=33785  reported for 5.0.54 fixed in 5.0 series at 5.0.87 
The values adjusted to are 4G::

    In [86]: ( 1 << 32 ) - 1
    Out[86]: 4294967295L

    In [21]: (4294967295+1)/1024/1024/1024
    Out[21]: 4L


From `Stage 2: Easy safe repair`

If you want a repair operation to go much faster, you should set the values of
the `sort_buffer_size` and `key_buffer_size` variables each to about 25% of your
available memory when running myisamchk.

Unfortunately I do not have 16G of memory, so potentially a repair will run out of memory with these settings.


After that succeed to create MYI of precisely the prior size, but different content::

    [root@belle7 tmp_offline_db_ext]# ll DqChannelPacked.MYI ../tmp_offline_db_ext_keep/DqChannelPacked.MYI
    -rw-rw---- 1 mysql mysql 4621312 May 21 13:46 ../tmp_offline_db_ext_keep/DqChannelPacked.MYI
    -rw-rw---- 1 mysql mysql 4621312 May 22 19:08 DqChannelPacked.MYI
    [root@belle7 tmp_offline_db_ext]# 
    [root@belle7 tmp_offline_db_ext]# diff -b  DqChannelPacked.MYI ../tmp_offline_db_ext_keep/DqChannelPacked.MYI
    Binary files DqChannelPacked.MYI and ../tmp_offline_db_ext_keep/DqChannelPacked.MYI differ
    [root@belle7 tmp_offline_db_ext]# 

Hexdump comparison::

    [root@belle7 tmp_offline_db_ext]# xxd -c 64 DqChannelPacked.MYI > /tmp/s/DqChannelPacked_MYI_recreated.xxd
    [root@belle7 tmp_offline_db_ext]# xxd -c 64 ../tmp_offline_db_ext_keep/DqChannelPacked.MYI > /tmp/s/DqChannelPacked_MYI_original.xxd

Shows differences only within first 4 lines of the dump (256 bytes of the MYI). Just header differences perhaps::

    [root@belle7 tmp_offline_db_ext]# diff  /tmp/s/DqChannelPacked_MYI_original.xxd /tmp/s/DqChannelPacked_MYI_recreated.xxd
    1,4c1,4
    < 0000000: fefe 0701 0000 01b0 00b0 0064 00c8 0002 0000 0100 0801 0000 0000 20ff 0000 0000 0004 edb8 0000 0000 0000 0000 0000 0000 0004 edb8 ffff ffff ffff ffff 0000 0000  ...........d.............. .....................................
    < 0000040: 0046 8400 0000 0000 00e2 b710 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 ef79 cdce 0000 615b 0000 0011 0000 0000 0000 0001 0000 0000  .F.......................................y....a[................
    < 0000080: 0046 8000 ffff ffff ffff ffff 0000 0000 0000 0000 519b 0976 0000 0000 0000 0001 0000 0000 519b 0975 0000 0000 0000 0000 0000 0000 519b 0a20 0000 0000 0004 edb8  .F..................Q..v............Q..u............Q.. ........
    < 00000c0: 0000 0001 0000 0001 0000 0000 0000 0400 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 002e 0000 002e 0000 002e  ................................................................
    ---
    > 0000000: fefe 0701 0000 01b0 00b0 0064 00c8 0002 0000 0100 0801 0000 0000 28ff 0000 0000 0004 edb8 0000 0000 0000 0000 0000 0000 0004 edb8 ffff ffff ffff ffff 0000 0000  ...........d..............(.....................................
    > 0000040: 0046 8400 0000 0000 00e2 b710 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1a2b 0000 0004 0000 0000 0000 0001 0000 0000  .F.............................................+................
    > 0000080: 0046 8000 ffff ffff ffff ffff 0000 0000 0000 0000 519c a52f 0000 0000 0000 0001 0000 0000 519c a52f 0000 0000 0000 0000 0000 0000 519c a74b 0000 0000 0004 edb8  .F..................Q../............Q../............Q..K........
    > 00000c0: 0000 0000 0000 0000 0000 0000 0000 0400 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 002e 0000 002e 0000 002e  ................................................................


Check the table survived this trauma
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Seems so::

    mysql> show tables ;
    +------------------------------+
    | Tables_in_tmp_offline_db_ext |
    +------------------------------+
    | DqChannelPacked              | 
    | DqChannelPackedVld           | 
    +------------------------------+
    2 rows in set (0.00 sec)

    mysql> select count(*) from DqChannelPacked   ;
    +----------+
    | count(*) |
    +----------+
    |   323000 | 
    +----------+
    1 row in set (0.00 sec)

    mysql> select * from DqChannelPacked where SEQNO=101010 ;
    +--------+-------------+-------+--------+------------+------------+------------+------------+------------+------------+-------+
    | SEQNO  | ROW_COUNTER | RUNNO | FILENO | MASK0      | MASK1      | MASK2      | MASK3      | MASK4      | MASK5      | MASK6 |
    +--------+-------------+-------+--------+------------+------------+------------+------------+------------+------------+-------+
    | 101010 |           1 | 21520 |    245 | 2147483647 | 2147483647 | 2147483647 | 2147483647 | 2147483647 | 2147483647 |    63 | 
    +--------+-------------+-------+--------+------------+------------+------------+------------+------------+------------+-------+
    1 row in set (0.00 sec)





myisamcheck memory
-------------------

* http://dev.mysql.com/doc/refman/5.0/en/myisamchk-memory.html

On N have 4G of memory, so need to restrict to 1G::

    [blyth@belle7 ~]$ free -m
                 total       used       free     shared    buffers     cached
    Mem:          4052       1639       2412          0        576        684
    -/+ buffers/cache:        378       3673
    Swap:         1983          0       1983

::

    myisamchk --sort_buffer_size=256M --key_buffer_size=512M --read_buffer_size=64M --write_buffer_size=64M       # suggestion for 512MB available
    myisamchk --sort_buffer_size=512M --key_buffer_size=1024M --read_buffer_size=128M --write_buffer_size=128M     # suggestion for 512MB available doubled
   

No speed difference, but maybe as nothing to fix::

    [root@belle7 tmp_offline_db_ext]# time myisamchk --sort_buffer_size=512M --key_buffer_size=1024M --read_buffer_size=128M --write_buffer_size=128M    -r -q DqChannelPacked
    Warning: option 'key_buffer_size': unsigned value 18446744073709551615 adjusted to 4294963200
    Warning: option 'read_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'write_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'sort_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    - check record delete-chain
    - recovering (with sort) MyISAM-table 'DqChannelPacked'
    Data records: 323000
    - Fixing index 1
              
    real    0m0.291s
    user    0m0.235s
    sys     0m0.050s

    [root@belle7 tmp_offline_db_ext]# time myisamchk  -r -q DqChannelPacked
    Warning: option 'key_buffer_size': unsigned value 18446744073709551615 adjusted to 4294963200
    Warning: option 'read_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'write_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'sort_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    - check record delete-chain
    - recovering (with sort) MyISAM-table 'DqChannelPacked'
    Data records: 323000
    - Fixing index 1
              
    real    0m0.290s
    user    0m0.242s
    sys     0m0.048s

Help variable dumping suggests it gets the message despite the warnings::

    [root@belle7 tmp_offline_db_ext]# myisamchk --help | grep size
    Warning: option 'key_buffer_size': unsigned value 18446744073709551615 adjusted to 4294963200
    Warning: option 'read_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'write_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'sort_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    key_buffer_size                   520192
    key_cache_block_size              1024
    myisam_block_size                 1024
    read_buffer_size                  262136
    write_buffer_size                 262136
    sort_buffer_size                  2097144
    [root@belle7 tmp_offline_db_ext]# 
    [root@belle7 tmp_offline_db_ext]# 
    [root@belle7 tmp_offline_db_ext]# myisamchk --sort_buffer_size=512M --key_buffer_size=1024M --read_buffer_size=128M --write_buffer_size=128M  | grep size
    Warning: option 'key_buffer_size': unsigned value 18446744073709551615 adjusted to 4294963200
    Warning: option 'read_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'write_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    Warning: option 'sort_buffer_size': unsigned value 18446744073709551615 adjusted to 4294967295
    key_buffer_size                   1073741824
    key_cache_block_size              1024
    myisam_block_size                 1024
    read_buffer_size                  134217728
    write_buffer_size                 134217728
    sort_buffer_size                  536870912
    [root@belle7 tmp_offline_db_ext]# 




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








