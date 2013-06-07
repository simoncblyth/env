Information Schema
====================

Updates and counts in all tables of all databases
---------------------------------------------------
::

    [blyth@belle7 ~]$ echo "select @@hostname,current_user(),concat(table_schema,'.',table_name),table_rows,create_time,update_time from information_schema.tables"  | dybdb1_ihep_ac_cn
    +-------------------+----------------+----------------------------------------------------------+------------+---------------------+---------------------+
    | @@hostname        | current_user() | concat(table_schema,'.',table_name)                      | table_rows | create_time         | update_time         |
    +-------------------+----------------+----------------------------------------------------------+------------+---------------------+---------------------+
    | dybdb1.ihep.ac.cn | dayabay@%      | information_schema.CHARACTER_SETS                        |       NULL | NULL                | NULL                | 
    | dybdb1.ihep.ac.cn | dayabay@%      | information_schema.COLLATIONS                            |       NULL | NULL                | NULL                | 
    ...
    | dybdb1.ihep.ac.cn | dayabay@%      | offline_db.AdWpHvMapVld                                  |         22 | 2012-08-25 14:14:55 | 2012-12-17 15:27:34 | 
    | dybdb1.ihep.ac.cn | dayabay@%      | offline_db.AdWpHvSetting                                 |      11328 | 2012-08-25 14:15:48 | 2013-06-01 11:18:18 | 
    | dybdb1.ihep.ac.cn | dayabay@%      | offline_db.AdWpHvSettingVld                              |         59 | 2012-08-25 14:15:48 | 2013-06-01 11:18:18 | 
    | dybdb1.ihep.ac.cn | dayabay@%      | offline_db.AdWpHvToFee                                   |       1536 | 2012-08-25 14:15:23 | 2012-08-30 20:21:34 | 
    | dybdb1.ihep.ac.cn | dayabay@%      | offline_db.AdWpHvToFeeVld                                |          8 | 2012-08-25 14:15:23 | 2012-08-30 20:21:34 | 
    | dybdb1.ihep.ac.cn | dayabay@%      | offline_db.CableMap                                      |     128885 | 2011-06-17 14:19:46 | 2012-05-04 17:14:31 | 



