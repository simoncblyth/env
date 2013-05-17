Locking used by mysqlhotcopy
================================ 


Following mysql log
--------------------

A default run of *mysqlhotcopy* results in the below activity in the mysql log, as followed by `mysql-tail`
::

    130516 12:03:24 2400342 Connect     root@localhost on information_schema
                    2400342 Query       set autocommit=1
                    2400342 Query       show variables like 'datadir'
                    2400342 Query       SHOW TABLES FROM `tmp_offline_db`
                    2400342 Query       LOCK TABLES `tmp_offline_db`.`CableMap` READ, 
                                                    `tmp_offline_db`.`CableMapVld` READ, 
                                                    ...
                    2400342 Query       FLUSH TABLES /\*!32323 `tmp_offline_db`.`CableMap`, 
                                                               `tmp_offline_db`.`CableMapVld`,
                                                      ... 


    130516 12:03:25 2400342 Query       UNLOCK TABLES
                    2400342 Quit       



Table locking
--------------

Checking the `/usr/bin/mysqlhotcopy` perl script source

The official docs are not very explicit, but imply:

   * with a READ lock, any connection can read from the table, but no connection can write to the table


#. `lock tables <name> READ`  
 
    * http://dev.mysql.com/doc/refman/5.0/en/lock-tables.html

    * A table lock protects only against inappropriate reads or writes by other sessions. 
      The session holding the lock, even a read lock, can perform table-level operations such as DROP TABLE. 

    * *READ* locks mean : 

         * The session that holds the lock can read the table (but not write it).
         * Multiple sessions can acquire a READ lock for the table at the same time.
         * Other sessions can read the table without explicitly acquiring a READ lock.

    * *WRITE* lock

         * The session that holds the lock can read and write the table.
         * Only the session that holds the lock can access the table. No other session can access it until the lock is released.
         * Lock requests for the table by other sessions block while the WRITE lock is held.


#. `flush tables` ensures whats is on disk is the latest state of the tables 

    * http://dev.mysql.com/doc/refman/5.0/en/flush.html
    * Closes all open tables, forces all tables in use to be closed, and flushes the query cache



From the first session::

    mysql> CREATE TABLE t1 (ID INT NOT NULL);
    Query OK, 0 rows affected (0.02 sec)

    mysql> INSERT INTO t1 VALUES (101);
    Query OK, 1 row affected (0.00 sec)

    mysql> LOCK TABLE t1 READ;
    Query OK, 0 rows affected (0.00 sec)

    mysql> INSERT INTO t1 VALUES (102);
    ERROR 1099 (HY000): Table 't1' was locked with a READ lock and can't be updated

    mysql> LOCK TABLE t1 WRITE;
    Query OK, 0 rows affected (0.00 sec)

    mysql> INSERT INTO t1 VALUES (102);
    Query OK, 1 row affected (0.00 sec)

    mysql> INSERT INTO t1 VALUES (103);
    Query OK, 1 row affected (0.00 sec)

    mysql> /* at this point startup another session and do a select, its blocked */

    mysql> LOCK TABLE t1 READ ;
    Query OK, 0 rows affected (0.00 sec)

    mysql> /* over in the other session downgrading the lock from WRITE to READ allowed the select to complete */

    mysql> /* an insert from the other session is hanging around blocked */

    mysql> UNLOCK TABLES ;
    Query OK, 0 rows affected (0.00 sec)

    mysql> /* the insert from the other session proceeded when the unlock was done */


Other session::

    mysql> select * from t1 ;
    +-----+
    | ID  |
    +-----+
    | 101 | 
    | 102 | 
    | 103 | 
    +-----+
    3 rows in set (1 min 9.29 sec)

    mysql> INSERT INTO t1 VALUES (201);
    Query OK, 1 row affected (3 min 40.96 sec)


Checking processlist while one session is locked out, and has hung around for 2607 seconds::

    mysql> show processlist ;
    +---------+------+-------------------------+----------------+---------+------+--------+-----------------------------+
    | Id      | User | Host                    | db             | Command | Time | State  | Info                        |
    +---------+------+-------------------------+----------------+---------+------+--------+-----------------------------+
    | 2400340 | root | belle7.nuu.edu.tw:34227 | tmp_offline_db | Sleep   | 2657 |        | NULL                        | 
    | 2400343 | root | belle7.nuu.edu.tw:34148 | tmp_offline_db | Query   |    0 | NULL   | show processlist            | 
    | 2400344 | root | belle7.nuu.edu.tw:49003 | tmp_offline_db | Query   | 2607 | Locked | INSERT INTO t1 VALUES (301) | 
    +---------+------+-------------------------+----------------+---------+------+--------+-----------------------------+
    3 rows in set (0.00 sec)


On accidentally closing the session that held the read lock the lock was releases and the other 
session proceeded with the insert after 46min of hanging around::

    mysql> INSERT INTO t1 VALUES (301);
    Query OK, 1 row affected (46 min 16.92 sec)




Locking Timeouts
^^^^^^^^^^^^^^^^^^^

::

    mysql> select @@hostname ;
    +-------------------+
    | @@hostname        |
    +-------------------+
    | belle7.nuu.edu.tw | 
    +-------------------+
    1 row in set (0.00 sec)

    mysql> show variables like '%timeout' ; 
    +----------------------------+-------+
    | Variable_name              | Value |
    +----------------------------+-------+
    | connect_timeout            | 10    | 
    | delayed_insert_timeout     | 300   | 
    | innodb_lock_wait_timeout   | 50    | 
    | innodb_rollback_on_timeout | OFF   | 
    | interactive_timeout        | 28800 | 
    | net_read_timeout           | 30    | 
    | net_write_timeout          | 60    | 
    | slave_net_timeout          | 3600  | 
    | table_lock_wait_timeout    | 50    | 
    | wait_timeout               | 28800 | 
    +----------------------------+-------+
    10 rows in set (0.00 sec)


Apparently `table_lock_wait_timeout` is unused, http://bugs.mysql.com/bug.php?id=41949 by observation the 
locks wait around much longer then 50s

The effective timeout maybe `wait_timeout` which defaults to 8 hrs
* http://dev.mysql.com/doc/refman/5.1/en/server-system-variables.html#sysvar_wait_timeout

::

    mysql> select @@hostname, @@wait_timeout/60/60 ;
    +-------------------+----------------------+
    | @@hostname        | @@wait_timeout/60/60 |
    +-------------------+----------------------+
    | belle7.nuu.edu.tw |           8.00000000 | 
    +-------------------+----------------------+
    1 row in set (0.00 sec)



External (file level locking)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

External locking is the use of file system locking to manage contention for MyISAM database tables by multiple processes. 

* http://dev.mysql.com/doc/refman/5.0/en/external-locking.html
* http://dev.mysql.com/doc/refman/5.0/en/myisamchk.html

With external locking disabled, to use myisamchk, you must either stop the
server while myisamchk executes or else lock and flush the tables before
running myisamchk. To avoid this requirement, use the CHECK TABLE and REPAIR TABLE
statements to check and repair MyISAM tables.

* *lock and flush is what hotcopy does*


::

    mysql> show variables like '%locking' ;
    +-----------------------+-------+
    | Variable_name         | Value |
    +-----------------------+-------+
    | skip_external_locking | ON    | 
    +-----------------------+-------+
    1 row in set (0.00 sec)

    mysql> select @@hostname ;
    +-------------------+
    | @@hostname        |
    +-------------------+
    | belle7.nuu.edu.tw | 
    +-------------------+
    1 row in set (0.00 sec)




