#!/usr/bin/env python
"""
Usage::

   export DBPATH=/data/env/tmp/tracs/dybsvn/2013/05/16/104702/dybsvn/db/trac.db
   ./db.py 

"""
import os, sys, logging
log = logging.getLogger(__name__)

try:
    import sqlite3 as sqlite
except ImportError:
    from pysqlite2 import dbapi2 as sqlite


def dict_factory(cursor, row):
    """
    Hookup to the connection with::
 
        conn.row_factory = dict_factory
        #conn.row_factory = sqlite.Row 

    The faster `Row` alternative has a pseudo-dict interface, 
    real dicts are more convenient for development.
    """ 
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


class DB(object):
    def __init__(self, path=None):
        """
        :param path:

        The `Row` class provides a dict like interface, but its not a dict
        """
        log.debug(self.versions())
        if not path:
            path = os.environ['DBPATH']
        log.debug("connecting to %s " % path )
        conn = sqlite.connect(path)
        conn.row_factory = dict_factory
        pass
        self.path = path
        self.conn = conn 

    def externally(self, sql):
        path = self.path
        print "\n"+os.popen("echo \"%(sql)s ; \" | sqlite3 -header -column  %(path)s " % locals()).read()  # with the sqlite3 binary 

    def __call__(self, sql ):
        log.debug(sql)
        conn = self.conn
        cursor = conn.cursor()
        return cursor.execute(sql)

    def versions(self):
        """
        * http://www.sqlite.org/changes.html

        2009-05-19 (3.6.14.1)  possibly obscure `group_concat` bug fixed http://www.sqlite.org/cvstrac/tktview?tn=3841 
        2007-12-14 (3.5.4) `group_concat` introduced 

        ::
 
            python -c "import platform, sys, sqlite3 ; print '%s %s %s' % ( platform.node(), '.'.join(map(str,sys.version_info[0:3])), sqlite3.sqlite_version ) "

        ::

            belle7.nuu.edu.tw       2.7.0   3.6.8        nuwa python
            simon.phys.ntu.edu.tw   2.5.6   3.7.14.1     macports 
            cms01.phys.ntu.edu.tw   2.5.1   3.1.2        source python
            cms01.phys.ntu.edu.tw   2.7.0   3.6.8        nuwa python

        """
        pyver = ".".join(map(str,sys.version_info[0:3]))
        return "python %s sqlite3.version = %s sqlite3.sqlite_version = %s " % ( pyver, sqlite.version, sqlite.sqlite_version )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    db = DB()
    for r in db("select * from bitten_build limit 10"):
        print r



