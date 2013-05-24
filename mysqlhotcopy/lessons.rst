
Lessons from MySQL corruption incident
========================================

Such rare events of DB corruption may recur no matter what we do.
The improvements we implement in response to this should focus on 
preventive measures, mitigating the pain/disruption if it does recur
and defining and documenting procedures for such incidents.

#. preventive avoidance 

   * avoid writing more that we need to, the DqChannelStatus 
     tables use a ruinously profligate schema (by a factor of **125**) 
     they are currently ~2350 MB (~14 months) they 
     could be ~19 MB with no loss of information. 
     As demonstrated by the size of DqChannelPacked. 

     The probability of corruption probably scales with the 
     time/volume of writes so it is no surprise that DQ tables
     are the first to see corruption.

   * disk space monitoring at least daily with email notifications 
     on all critical nodes especially `dybdb1.ihep.ac.cn` and `dybdb2.ihep.ac.cn`, 
     reorganization/purchases to avoid tight disk space 

   * queue draining procedures for DB writing jobs


#. mitigating the pain/disruption 

   * automatic daily backups and remote tarball transfers of all critical databases, 
     ("offline_db" is already, "tmp_ligs_offline_db" is not currently and its size
     will make this difficult)
     Replication is not a replacement for backups as "drop database" gets 
     propagated along the chain within seconds.
     
     The servers are claimed to have disk level backups. However these do not 
     lock the DB during the backup and without regular tests that backups
     are recoverable from I do not trust them.  The backups of offline_db are 
     recovered onto an NTU node daily. 

   * DBI writing into a "crashed" payload table should have caused DBI 
     to refuse writing into any table and exit the jobs abruptly. 
     This would have given a clean cut, and hopefully notified us of the 
     problem sooner.
     
     I have the crashed table in a tarball, so I can now reproduce  
     DBI writing into a crashed table and observe the current error handling 
     and see where improvements need to be made. Either in DBI/DybDBI or its 
     usage/documentation. Perhaps just need all unattended writing to check
     written SEQNO::

          wseqno = wrt.Close()  # closing DybDbi writer returns SEQNO written OR zero on error
          assert wseqno, wseqno

   * processes that perform critical tasks such as DB writing need logfile monitoring 
     with email notifications when things go awry


#. proto-SOP for MySQL corruption

   * stop writing to corrupted tables as soon as detected, and inhibit future writing to related tables
     until recovery is done and resumption is agreed by DB managers, sys admins, KUP job admins
     and table experts
     
      * trying to allow writing to continue is pointless, 
        as this just creates mess that will needs to removed anyhow

      * need simple/reliable way to inhibit DB writing for a set of tables,  
        possibly just need to find/test the appropriate `GRANT SELECT ON ...` 
        to make tables read only for all users other than root.

   * perform mysqldump (possibly with some SEQNO excluded) or mysqlhotcopy and 
     transfer to a remote node in which the they are recovered from









::

    mysql> select concat(table_schema,".",table_name),table_type, engine, round((data_length+index_length-data_free)/1024/1024,2) as MB  from information_schema.tables where substr(table_name,1,2) = 'Dq' ;
    +------------------------------------------+------------+--------+---------+
    | concat(table_schema,".",table_name)      | table_type | engine | MB      |
    +------------------------------------------+------------+--------+---------+
    | tmp_ligs_offline_db_0.DqChannelStatus    | BASE TABLE | MyISAM | 2265.14 | 
    | tmp_ligs_offline_db_0.DqChannelStatusVld | BASE TABLE | MyISAM |   20.24 | 
    | tmp_ligs_offline_db_1.DqChannelStatus    | BASE TABLE | MyISAM | 2349.86 | 
    | tmp_ligs_offline_db_1.DqChannelStatusVld | BASE TABLE | MyISAM |   20.24 | 
    | tmp_offline_db.DqChannelPacked           | BASE TABLE | MyISAM |   18.61 | 
    | tmp_offline_db.DqChannelPackedVld        | BASE TABLE | MyISAM |   18.87 | 
    +------------------------------------------+------------+--------+---------+
    6 rows in set (0.01 sec)

    mysql> select max(SEQNO) from tmp_offline_db.DqChannelPacked ;
    +------------+
    | max(SEQNO) |
    +------------+
    |     323000 | 
    +------------+
    1 row in set (0.04 sec)

    mysql> select max(SEQNO) from tmp_ligs_offline_db_1.DqChannelStatus ;
    +------------+
    | max(SEQNO) |
    +------------+
    |     340817 | 
    +------------+
    1 row in set (0.06 sec)

    mysql> select 2349.86/18.61 ;
    +---------------+
    | 2349.86/18.61 |
    +---------------+
    |    126.268673 | 
    +---------------+
    1 row in set (0.00 sec)

