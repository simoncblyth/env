#!/usr/bin/env python
"""
Usage::

   export DBPATH=/data/env/tmp/tracs/dybsvn/2013/05/16/104702/dybsvn/db/trac.db
   ./db.py 

"""
import os, logging
log = logging.getLogger(__name__)
import sqlite3 


def dict_factory(cursor, row):
    """
    Hookup to the connection with::
 
        conn.row_factory = dict_factory
        #conn.row_factory = sqlite3.Row 

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
        if not path:
            path = os.environ['DBPATH']
        log.info("connecting to %s " % path )
        conn = sqlite3.connect(path)
        conn.row_factory = dict_factory
        pass
        self.path = path
        self.conn = conn 

    def externally(self, sql):
        path = self.path
        print "\n"+os.popen("echo \"%(sql)s ; \" | sqlite3 -header -column  %(path)s " % locals()).read()  # with the sqlite3 binary 

    def __call__(self, sql ):
        log.info(sql)
        conn = self.conn
        cursor = conn.cursor()
        return cursor.execute(sql)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    db = DB()
    for r in db("select * from bitten_build limit 10"):
        print r



