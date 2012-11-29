Shrinkage
==========

.. contents:: :local:

Fat Trac DB 
-------------

The Trac DB needs to be shrunk, currently at 5.5G 
Extract trac.db from tarball and interactively check with::

	[dayabay] /tmp/tt > tar zxf dybsvn.tar.gz dybsvn/db/trac.db
	[dayabay] /tmp/tt > du -h dybsvn/db/trac.db
	5.5G    dybsvn/db/trac.db
	[dayabay] /tmp/tt > sqlite3 dybsvn/db/trac.db
	SQLite version 3.3.3
	Enter ".help" for instructions
	sqlite> .tables
	attachment          bitten_report       node_change         ticket
	auth_cookie         bitten_report_item  permission          ticket_change
	bitten_build        bitten_rule         report              ticket_custom
	bitten_config       bitten_slave        revision            version
	bitten_error        bitten_step         session             wiki
	bitten_log          component           session_attribute
	bitten_log_message  enum                system
	bitten_platform     milestone           tags
	sqlite> 


Which tables hold the fat::

        ./count.py /tmp/tt/dybsvn/db/trac.db

	bitten_config                  : 5 
	bitten_platform                : 16 
	bitten_rule                    : 16 
	bitten_error                   : 8164 
	bitten_build                   : 13170 
	bitten_report                  : 98520 
	bitten_slave                   : 162784 
	bitten_log                     : 298469 
	bitten_step                    : 300033 
	bitten_report_item             : 18051235     ## more than 18M 
	bitten_log_message             : 39656123     ## almost 40M 

Schema of fatties::


	sqlite> .schema bitten_log_message
	CREATE TABLE bitten_log_message (
	    log integer,
	    line integer,
	    level text,
	    message text,
	    UNIQUE (log,line)
	);

	sqlite> .schema bitten_report_item  
	CREATE TABLE bitten_report_item (
	    report integer,
	    item integer,
	    name text,
	    value text,
	    UNIQUE (report,item,name)
	);

	sqlite> select max(length(message)) from bitten_log_message ;
	3571

	sqlite> select sum(length(message)) from bitten_log_message ;
	2048409596      ## 2 billion chars of log messages


Work out the query::


	sqlite> select count(*) FROM bitten_log_message WHERE log < (SELECT max(id) FROM bitten_log WHERE build < 10000 ) ;
	10478700

	sqlite> SELECT max(id) FROM bitten_log WHERE build < 10000 ;
	148781

	sqlite> select count(*) FROM bitten_log_message ;
	39656123
	sqlite> 

	sqlite> select cast(rev as integer) from bitten_build limit 10 ;
	3953
	3952
	3957

	sqlite> select count(*) from bitten_build where cast(rev as int) < 10000 ;
	3175

	sqlite> select count(distinct(id)) from bitten_build where cast(rev as int) < 10000 ;
	3175

	sqlite> select count(distinct(id)) from bitten_build where cast(rev as int) > 10000 ;
	9995

	sqlite> SELECT min(rev+0),max(rev+0) FROM bitten_build  ;
	3952|17979

Hmm will killing all of a configs builds cause problems for Trac/Bitten web interface::


	sqlite> select distinct(config) from bitten_build;
	detdesc
	dybinst
	dybdoc
	dybdaily
	opt.dybinst



How to slim 
---------------

The 10 slaves are calling home every 5 minutes so lots of contention potential 
for what is probably an expensive sequence of deletes, and the probable vacuuming. 
Slave death during this operation would not be surprising : however their supervisord
should auto-restart them.

  * need to lock ? like trac admin hotcopy does 
    
       * nope : locking would mean cannot do the deletes, whats needed is to do a long running sequence of several queries transactionally 
       * http://www.sqlite.org/transactional.html
       * http://www.sqlite.org/atomiccommit.html  
       * http://www.sqlite.org/lockingv3.html  

  * best to do via a patch to tracadmin that allows arbitary sql to be run agains the DB in a lock/unlock manner
             
        * /data/env/local/env/trac/package/tractrac/trac-0.11/trac/admin/console.py do_hotcopy 
        * create a do_arbitary following example of do_hotcopy that runs arbitary sql from a path given

  * peer into the future of trac/bitten to see if such things are implemented

        * http://trac.edgewall.org/browser/trunk/trac/admin/console.py  a great deal of divergence 
        * http://trac.edgewall.org/log/trunk/trac/admin/console.py 

  * open question : maybe need to vacuum sqlite DB for size decrease after deletions are done

        * http://www.sqlite.org/lang_vacuum.html  needs twice DB size in free space 



Adding arbitary tracadmin command
-----------------------------------

Need a backup trac environment dir to test, just the db is insufficient as
need the config file.

::

    [blyth@cms02 trac-0.11]$ TRAC_INSTANCE=toysvn trac-admin-


Hmm difficult develop at trac admin level first, so do at pysqlite level to allow practicing on a copy of the DB.


Transactions at pysqlite level
--------------------------------

But pysqlite already using this it seems.::

        [blyth@cms02 trac-0.11]$ which trac-admin
        /data/env/system/python/Python-2.5.6/bin/trac-admin

 * :google:`pysqlite try except finally rollback commit`


Transaction Control At The SQL Level
---------------------------------------

Quoting from  http://www.sqlite.org/lockingv3.html


The changes to locking and concurrency control in SQLite version 3 also
introduce some subtle changes in the way transactions work at the SQL language
level. By default, SQLite version 3 operates in autocommit mode. In autocommit
mode, all changes to the database are committed as soon as all operations
associated with the current database connection complete.

The SQL command "BEGIN TRANSACTION" (the TRANSACTION keyword is optional) is
used to take SQLite out of autocommit mode. Note that the BEGIN command does
not acquire any locks on the database. After a BEGIN command, a SHARED lock
will be acquired when the first SELECT statement is executed. A RESERVED lock
will be acquired when the first INSERT, UPDATE, or DELETE statement is
executed. No EXCLUSIVE lock is acquired until either the memory cache fills up
and must be spilled to disk or until the transaction commits. In this way, the
system delays blocking read access to the file file until the last possible
moment.

The SQL command "COMMIT" does not actually commit the changes to disk. It just
turns autocommit back on. Then, at the conclusion of the command, the regular
autocommit logic takes over and causes the actual commit to disk to occur. The
SQL command "ROLLBACK" also operates by turning autocommit back on, but it also
sets a flag that tells the autocommit logic to rollback rather than commit.

If the SQL COMMIT command turns autocommit on and the autocommit logic then
tries to commit change but fails because some other process is holding a SHARED
lock, then autocommit is turned back off automatically. This allows the user to
retry the COMMIT at a later time after the SHARED lock has had an opportunity
to clear.

If multiple commands are being executed against the same SQLite database
connection at the same time, the autocommit is deferred until the very last
command completes. For example, if a SELECT statement is being executed, the
execution of the command will pause as each row of the result is returned.
During this pause other INSERT, UPDATE, or DELETE commands can be executed
against other tables in the database. But none of these changes will commit
until the original SELECT statement finishes.


Using SQLite transactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  * http://www.sqlite.org/lang_transaction.html

::

       BEGIN EXCLUSIVE TRANSACTION ;
           select ...
       COMMIT TRANSACTION ;







dev shrink sql 
----------------

Objective : come up with SQL that does the desired shrinkage 

 * env/sqlite/shrink.py 

dev do_arbitary on C2
----------------------

Objective : add arbitary command allowing execution of arbitary SQL from a file specified from trac admin console or command line with locking/unlocking based on the hotcopy command 

Get there::

      tractrac-
      tractrac-cd
      vi trac/admin/console.py

No motivation to stuff into git for cleaner maintenance that current patch approach as this my first change to console.py::

        [blyth@cms02 trac-0.11]$ svn st trac/admin/console.py 
        [blyth@cms02 trac-0.11]$ svn log trac/admin/console.py 
        ------------------------------------------------------------------------
        r7236 | jonas | 2008-06-22 23:43:50 +0800 (Sun, 22 Jun 2008) | 1 line

        Tagging trac-0.11
        ------------------------------------------------------------------------
        r6940 | jonas | 2008-05-01 01:44:57 +0800 (Thu, 01 May 2008) | 1 line

        Creating branch 0.11-stable
        ------------------------------------------------------------------------




Caution uncommitted on C2
---------------------------


::

        [blyth@cms02 trac-0.11]$ svn diff trac/admin/console.py
        Index: trac/admin/console.py
        ===================================================================
        --- trac/admin/console.py       (revision 7236)
        +++ trac/admin/console.py       (working copy)
        @@ -254,7 +254,7 @@
             _help_help = [('help', 'Show documentation')]
         
             def all_docs(cls):
        -        return (cls._help_help + cls._help_initenv + cls._help_hotcopy +
        +        return (cls._help_help + cls._help_initenv + cls._help_hotcopy + cls._help_arbitary +
                         cls._help_resync + cls._help_upgrade + cls._help_deploy +
                         cls._help_permission + cls._help_wiki +
                         cls._help_ticket + cls._help_ticket_type + 
        @@ -1151,6 +1151,35 @@
         
                 print 'Hotcopy done.'
         
        +    _help_arbitary = [('arbitary <path/to/sql/script>',
        +                      'Run arbitary sql script against the SQLite DB (expensive/dangerous query sequences should use transactions)')]
        +    def do_arbitary(self, line):
        +        arg = self.arg_tokenize(line)
        +        if arg[0]:
        +            script = arg[0]
        +        else:
        +            self.do_help('arbitary')
        +            return
        +
        +        if not os.path.exists(script):
        +            raise TracError(_("arbitary can't read script '%(script)s'",
        +                              script=script))
        +
        +        sql = open(script,"r").read()
        +        cnx = self.db_open()
        +        cursor = cnx.cursor()
        +        try:
        +            print 'Running arbitary script %s sql %s against %s ...' % (script, sql, self.__env.path),
        +            self.db_update(sql, cursor)
        +        except:
        +            cursor.close()
        +            cnx.rollback()
        +        finally:
        +            cnx.commit()
        +        print 'Arbitary done.'
        +
        +
        + 
             _help_deploy = [('deploy <directory>',
                              'Extract static resources from Trac and all plugins.')]
        



testing on C2
---------------


::

        [root@cms02 dayabay]# pwd
        /var/scm/backup/dayabay
        [root@cms02 dayabay]# mkdir -p tracs/dybsvn/2012/09/18/120001
        [root@cms02 dayabay]# scp WW:/home/scm/backup/dayabay/tracs/dybsvn/2012/09/18/120001/dybsvn.tar.gz.dna tracs/dybsvn/2012/09/18/120001/
        [root@cms02 dayabay]# screen time scp WW:/home/scm/backup/dayabay/tracs/dybsvn/2012/09/18/120001/dybsvn.tar.gz tracs/dybsvn/2012/09/18/120001/
        [screen is terminating]
        [root@cms02 dayabay]# time screen scp WW:/home/scm/backup/dayabay/svn/dybsvn/2012/09/18/120001/dybsvn-18208.tar.gz svn/dybsvn/2012/09/18/120001/
        [screen is terminating]
        real    44m12.252s
        user    0m0.001s
        sys     0m0.003s






