#!/usr/bin/env python
"""

Usage::

   ~/e/sqlite/db.py /tmp/env/blyth/shrink/dybsvn/db/trac.db


General purpose pysqlite access to SQLite DB, put schema specifics 
in other modules keeping this general purpose.

TODO:

* adopt a DBCONF equivalent approach for SQLite DB  



Check the encoding of sqlite db with::

    sqlite> PRAGMA encoding; 
    encoding  
    ----------
    UTF-8     


* https://stackoverflow.com/questions/2392732/sqlite-python-unicode-and-non-utf-data

Let me first state the goal as I understand it. The goal in processing various
encodings, if you are trying to convert between them, is to understand what
your source encoding is, then convert it to unicode using that source encoding,
then convert it to your desired encoding. Unicode is a base and encodings are
mappings of subsets of that base. utf_8 has room for every character in
unicode, but because they aren't in the same place as, for instance, latin_1, a
string encoded in utf_8 and sent to a latin_1 console will not look the way you
expect. In python the process of getting to unicode and into another encoding
looks like:

"""

try:
    import pysqlite2.dbapi2 as sqlite
    have_pysqlite = 2
except ImportError:
    try:
        import sqlite3 as sqlite
        have_pysqlite = 2
    except ImportError:
        try:
            import sqlite
            have_pysqlite = 1
        except ImportError:
            have_pysqlite = 0



import logging, sys
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
        conn=sqlite.connect(path)
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
        return (sqlite.version, sqlite.sqlite_version )

    def __call__(self, sql):
        return self.fetchall(sql)

    def line_by_line(self, path ):
        fp = open(path,"r")
        for line in fp.readlines():
            line = line.strip()
            if not line:continue
            assert line[-1] == ";" , ("expecting simple semicolon terminated sql statements on each non-blank line ", line)
            print "[%s]" % line 
            ret = self(line)
            print ret 
        fp.close()

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
        except sqlite.Error:
            cnx.rollback()
            raise
        else:
            cnx.commit()   # Error from this commit are not caught 
        cur.close()



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    n = len(sys.argv)
    if n>1:
        path = sys.argv[1] 
    else:
        path = None 
    if n > 2:
        script = sys.argv[2] 
    else:
        script = None

    db = DB(path, skip='bitten')
    print db.versions()

    print db.tables
    if script:
        db.arbitary_(script) 
    else:
        cnt = db.count_()
        print "\n".join(["%-30s : %s " % (t, cnt[t]) for t in sorted(cnt, key=lambda _:cnt[_])])



