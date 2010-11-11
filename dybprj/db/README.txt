= DB app : high level MySQL permissions controller =

  * what mapping granularity between these and the MySQL table perms ?
  * very high level best for pre-cooked usability from admin  
     * ... but what about ease of introspection during the bootstrap ?


== more general viz than vDBI, eg histos, vld sketch, ... ==

  Investigating using approach  ...
     * env.offline.dbn.DBn  connects to DB and pulls out numpy arrays 
     * vizualized numpy arrays with  matplotlib 
     * save into django response as pdf/png/svg

== Drilling / Selection interface ==


Ideas :




== Generic plots ==

Ideas 
  * database level : pie chart showing the row counts of each table
  * table level : big n-by-n plot of each columns histograms
  * column level : simple hist  

  * DybDbi live (or canned) query  at timestamp you click on 
    ... highlighting the resuling seqno   
     * add a DbiQuery to the model ?



== Custom (table/column specific) plots ==

Custom plots use known relationships between fields/tables 
in order to inform the plotting approach, in contract to generic
plots which have little understanding of what the data is representing.

==  Organization of table/column specific plots 

     * want to enable iteractively developable matplotlib 
       plots to be integrated into webapp by simply 
       adding the plotname.py into a table named folder in repo    

     * this will enable people to develop and add custom plots 
       to webapp without needing to understand webapp machinery,
       and avoid me having to create too many custom plots 

== Issues ==

Approach is using many DB connections ...
     * will probably need to go to scraped/cached approach 
     * unknown complications regarding multiple DB connections 

     * every comment in the list is causing a user lookup 


    * string / VARCHAR ?columns cause errors..
http://cms01.phys.ntu.edu.tw:8000/db/prior/SimPmtSpec/PMTDESCRIB/column.svg
     cannot perform reduce with flexible type

== model ==
  
   * User, Group  : django standard     
   * Permissions  :  use guardian?

   * Database
   * Table
     * name
     * writers ... User 
     * related-tables ?
   * Log 

=== Questions ===

  * guardian object perms ... are generically related to the objects 
    * will that work with the many-to-many situation of Users and Tables 

  * Logging : 
    * the admin logs activity ... how to reuse that ? 

=== Decisions... ===

  * '''not''' DBI specific, just treat *Vld and DbiLogEntry as other related tables  



== PERMISSIONS IMPLEMENTATION APPROACHES ==

  * Use django Permissions ... model (not instance) based 
     * means cannot have a Table model ... 
     * requires dynamic introspection (i did this before)
     * django does not support compound PK using in DBI tables (SEQNO,ROW_COUNTER)
        * when just using django models to hold permissions ... probably not a problem  

     * FORCES HOOP JUMPING FOR DYNAMIC SYNCDB 

  * Role-own per-instance permissions ... aping unix  

  * adopt django add-on that provides object permissions 

      * django-authority (the way to go pre django 1.2)
         * http://packages.python.org/django-authority/create_per_object_permission.html
         * http://groups.google.com/group/django-authority/browse_thread/thread/59ef512046ad41b3
      * BUT django 1.2 adds new possibilities
         * http://djangoadvent.com/1.2/object-permissions/ 
            * nice API
         * http://packages.python.org/django-guardian/
            * http://packages.python.org/django-guardian/assign.html

         * http://packages.python.org/django-permissions
            * put off by ugly documentation, and "role" usag ... non-agnostic seemingly 

   * and the winner is : '''django-guardian'''
       * http://packages.python.org/django-guardian/configuration.html


== how/when to sync with tables in offline_db ? ==

   * via an action ... actions need a selection 
       * http://code.djangoproject.com/ticket/10768

  * so implement an object to represent a database , and put the action on that model ...  
      * populate that from the settings.DATABASES list 
      * how to get to the cursors with multi-db 
    
== ugo (user-group-other) app ==

    * generic thinking : 
       * control actions on resources (such as Databases, Tables ) 
       * aping unix permissions ? user-group-other

    * in particular ...
        * provide high level control of offline_db Table permissions
           * ie respond to logged changes to a high level model (ie User added to Group with write permission for Table )  
             issue the needed GRANTS/REVOKES to the offline_db
        * user profiles 
           * view group memberships and permissions of User 
 
= NEXT STEPS ... =

 * how to capture permission groups for DBI table pair  updating 
 * introspect current MySQL GRANTS ?
     * http://dev.mysql.com/doc/refman/5.1/en/privilege-system.html
{{{
SHOW GRANTS FOR 'joe'@'office.example.com';
}}}
   * not easily parsible 

{{{
mysql> grant select on testdb_20101020.SimPmtSpec to dayabay@belle7.nuu.edu.tw ;
Query OK, 0 rows affected (0.03 sec)

mysql> grant update,delete on testdb_20101020.SimPmtSpec to dayabay@belle7.nuu.edu.tw ;
Query OK, 0 rows affected (0.00 sec)


}}}

{{{
mysql> select * from mysql.tables_priv ;
+-------------------+-----------------+---------+------------+----------------------------+---------------------+----------------------+-------------+
| Host              | Db              | User    | Table_name | Grantor                    | Timestamp           | Table_priv           | Column_priv |
+-------------------+-----------------+---------+------------+----------------------------+---------------------+----------------------+-------------+
| belle7.nuu.edu.tw | testdb_20101020 | dayabay | SimPmtSpec | root@cms01.phys.ntu.edu.tw | 2010-10-20 19:43:54 | Select,Update,Delete |             |
+-------------------+-----------------+---------+------------+----------------------------+---------------------+----------------------+-------------+
1 row in set (0.00 sec)

mysql> describe mysql.tables_priv ;
+-------------+-----------------------------------------------------------------------------------------------+------+-----+-------------------+-------+
| Field       | Type                                                                                          | Null | Key | Default           | Extra |
+-------------+-----------------------------------------------------------------------------------------------+------+-----+-------------------+-------+
| Host        | char(60)                                                                                      |      | PRI |                   |       |
| Db          | char(64)                                                                                      |      | PRI |                   |       |
| User        | char(16)                                                                                      |      | PRI |                   |       |
| Table_name  | char(64)                                                                                      |      | PRI |                   |       |
| Grantor     | char(77)                                                                                      |      | MUL |                   |       |
| Timestamp   | timestamp                                                                                     | YES  |     | CURRENT_TIMESTAMP |       |
| Table_priv  | set('Select','Insert','Update','Delete','Create','Drop','Grant','References','Index','Alter') |      |     |                   |       |
| Column_priv | set('Select','Insert','Update','References')                                                  |      |     |                   |       |
+-------------+-----------------------------------------------------------------------------------------------+------+-----+-------------------+-------+
8 rows in set (0.00 sec)

}}}


{{{
mysql> show grants for dayabay@belle7.nuu.edu.tw ;
+-------------------------------------------------------------------------------------------------+
| Grants for dayabay@belle7.nuu.edu.tw                                                            |
+-------------------------------------------------------------------------------------------------+
| GRANT USAGE ON *.* TO 'dayabay'@'belle7.nuu.edu.tw'                                             |
| GRANT SELECT, UPDATE, DELETE ON `testdb_20101020`.`SimPmtSpec` TO 'dayabay'@'belle7.nuu.edu.tw' |
+-------------------------------------------------------------------------------------------------+
2 rows in set (0.00 sec)
}}}



{{{
mysql> select * from mysql.tables_priv ;
Empty set (0.01 sec)
}}}


