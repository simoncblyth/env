#!/usr/bin/env python
"""

Usage::

   ~/e/sqlite/db.py /tmp/env/blyth/shrink/dybsvn/db/trac.db


General purpose pysqlite access to SQLite DB, put schema specifics 
in other modules keeping this general purpose.

TODO:

* adopt a DBCONF equivalent approach for SQLite DB  


"""
from __future__ import with_statement
import logging, sys, sqlite3
log = logging.getLogger(__name__)

class DB(object):
    def __init__(self, path=None, skip=None):
        """ 
        :param path: to DB file
        :param skip: prefix of table names to not include in counting, for faster testing
        """ 
        if not path:
            path = 'dybsvn/db/trac.db'   # specific leaking in as default

        log.info("opening %s " % path )
        conn=sqlite3.connect(path)
        cur=conn.cursor()
        self.conn = conn
        self.path = path
        self.skip = skip
        self.count = -1
        self.tables = self.tables_() 

    def versions(self):
        """
        :return: pysqlite version, underlying sqlite version
        """ 
        #return (sqlite3.version, sqlite3._sqlite.sqlite_version())
        return (sqlite3.version, sqlite3.sqlite_version )

    def __call__(self, sql):
        return self.fetchall(sql)

    def line_by_line(self, path ):
        with open(path,"r") as fp:
            for line in fp.readlines():
                line = line.strip()
                if not line:continue
                assert line[-1] == ";" , ("expecting simple semicolon terminated sql statements on each non-blank line ", line)
                print "[%s]" % line 
                ret = self(line)
                print ret 

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
            if self.skip and tab.startswith(self.skip):continue  # skip biggies for speed
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

    def arbitary_(self, sqlpath ):
        """
        http://trac.edgewall.org/pysqlite.org-mirror/ticket/259

        How to test the corners of errorspace here ?
        How to test in contentious environment ?
        """
        sql = open(sqlpath,"r").read()
        cnx = self.db_open()
        cur = cnx.cursor()
        try:
            print 'Running arbitary script %s sql %s against %s ...' % (sqlpath, sql, self.path),
            cur.executescript(sql)
        except sqlite3.Error:
            cnx.rollback()
            raise
        else:
            cnx.commit()   # Error from this commit are not caught 
        finally:           # a SyntaxError here indicates you are using an old python, try python- to pick up source one
            cur.close()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    n = len(sys.argv)
    path = sys.argv[1] if n > 1 else None 
    script = sys.argv[2] if n > 2 else None

    db = DB(path, skip='bitten')
    print db.versions()

    print db.tables
    if script:
        db.arbitary_(script) 
    else:
        cnt = db.count_()
        print "\n".join(["%-30s : %s " % (t, cnt[t]) for t in sorted(cnt, key=lambda _:cnt[_])])



