#!/usr/bin/env python
"""

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

Use this script to run some queries on the sqlite db::

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

sqlite> .schema bitten_log_message
CREATE TABLE bitten_log_message (
    log integer,
    line integer,
    level text,
    message text,
    UNIQUE (log,line)
);

sqlite> select max(length(message)) from bitten_log_message ;
3571

sqlite> select sum(length(message)) from bitten_log_message ;
2048409596
                            ## 2 billion chars of log messages


sqlite> .schema bitten_report_item  
CREATE TABLE bitten_report_item (
    report integer,
    item integer,
    name text,
    value text,
    UNIQUE (report,item,name)
);












The build id are monotonic with rev (or nearly so) : can just kill up to a certain manually determined build id

DELETE FROM bitten_build  WHERE id < 10000 

DELETE FROM bitten_error  WHERE build < 10000 
DELETE FROM bitten_step   WHERE build < 10000 
DELETE FROM bitten_slave  WHERE build < 10000 

DELETE FROM bitten_log     WHERE build < 10000 
DELETE FROM bitten_report  WHERE build < 10000 


DELETE
select count(*) FROM bitten_log_message WHERE log < (SELECT max(id) FROM bitten_log WHERE build < 10000 )


sqlite> select count(*) FROM bitten_log_message WHERE log < (SELECT max(id) FROM bitten_log WHERE build < 10000 ) ;
10478700
sqlite> SELECT max(id) FROM bitten_log WHERE build < 10000 ;
148781
sqlite> select count(*) FROM bitten_log_message ;
39656123
sqlite> 


select count(*) FROM bitten_report_item WHERE report < (SELECT max(id) FROM bitten_report WHERE build < 10000 )










sqlite> .schema bitten_build    
CREATE TABLE bitten_build (
    id integer PRIMARY KEY,
    config text,
    rev text,
    rev_time integer,
    platform integer,
    slave text,
    started integer,
    stopped integer,
    status text
);
CREATE INDEX bitten_build_config_rev_slave_idx ON bitten_build (config,rev,slave);
sqlite> 


   The rev is text, to support git and others with digest revisions

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


sqlite> SELECT count(*) FROM bitten_build where rev+0 < 10000 ;        # is there a better way to coerce 
3175
sqlite> SELECT count(distinct(id)) FROM bitten_build where rev+0 < 10000 ;
3175


sqlite> SELECT count(*) FROM bitten_build where rev+0 > 10000 ;
9995
sqlite> SELECT count(distinct(id)) FROM bitten_build where rev+0 > 10000 ;
9995










DELETE FROM bitten_log_message WHERE log IN (SELECT id FROM bitten_log WHERE build IN (SELECT id FROM bitten_build WHERE rev < 23000 AND config = 'trunk'))
DELETE FROM bitten_log WHERE build IN (SELECT id FROM bitten_build WHERE rev < 23000 AND config = 'trunk')
DELETE FROM bitten_error WHERE build IN (SELECT id FROM bitten_build WHERE rev < 23000 AND config = 'trunk')
DELETE FROM bitten_step WHERE build IN (SELECT id FROM bitten_build WHERE rev < 23000 AND config = 'trunk')
DELETE FROM bitten_slave WHERE build IN (SELECT id FROM bitten_build WHERE rev < 23000 AND config = 'trunk')
DELETE FROM bitten_build WHERE rev < 23000 AND config = 'trunk'




sqlite> select distinct(config) from bitten_build;
detdesc
dybinst
dybdoc
dybdaily
opt.dybinst



"""
import logging, sys, sqlite3
log = logging.getLogger(__name__)


class DB(object):
    def __init__(self, path=None):
        if not path:
            path = 'dybsvn/db/trac.db'
        conn=sqlite3.connect(path)
        cur=conn.cursor()
        self.conn = conn
        self.path = path
        self.count = -1
        self.tables = self.tables_() 

    def __call__(self, sql):
        return self.fetchall(sql)

    def execute_(self, sql):
        cursor = self.conn.cursor()
        cursor.execute( sql )
        return cursor

    def fetchall(self, sql ):
        log.debug(sql)
        cursor = self.execute_(sql)
        rows = cursor.fetchall()
        self.count = cursor.rowcount
        cursor.close()
        return rows

    def tables_(self):
        tabs = []
        for _ in self("select name from sqlite_master where type='table'"):
            tabs.append(_[0])
        return tabs

    def count_(self):
        """
        Initially return a list of tuples, requiring double indexing to get to the count
        """
        count={}
        for tab in self.tables:
            if tab.startswith('bitten'):continue  # skip biggies for speed
            sql = "select count(*) from %(tab)s" % locals()
            ret = self(sql)
            assert len(ret) == 1, ret
            ret = ret[0]
            assert len(ret) == 1, ret
            count[tab] = ret[0] 
        return count


    def db_open(self):
        """
        standin for trac/admin/console API
        """ 
        return self.conn

    def arbitary_(self, path ):
        """
        http://trac.edgewall.org/pysqlite.org-mirror/ticket/259

        How to test the corners of errorspace here ?
        How to test in contentious environment ?
        """
        sql = open(path,"r").read()
        cnx = self.db_open()
        cur = cnx.cursor()
        try:
            print 'Running arbitary script %s sql %s against %s ...' % (script, sql, self.path),
            cur.executescript(sql)
        except sqlite3.Error:
            cnx.rollback()
            raise
        else:
            cnx.commit()   # Error from this commit are not caught 
        finally:
            cur.close()


    def builds_(self, where="cast(rev as int) < 10000"):
        sql = "select id from bitten_build where %(where)s " % locals()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    n = len(sys.argv)
    path = sys.argv[1] if n > 1 else None 
    script = sys.argv[2] if n > 2 else None
    db = DB(path)
    print db.tables
    if script:
        db.arbitary_(script) 
    else:
        cnt = db.count_()
        print "\n".join(["%-30s : %s " % (t, cnt[t]) for t in sorted(cnt, key=lambda _:cnt[_])])



