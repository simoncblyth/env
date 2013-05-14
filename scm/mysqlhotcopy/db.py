#!/usr/bin/env python
"""

"""
import os, logging
import MySQLdb
from ConfigParser import ConfigParser
log = logging.getLogger(__name__)

class DB(object):
    def __init__(self, sect):
        """
        :param sect: used as the `read_default_group` in MySQLdb connection 
        """
        self.sect = sect
        self.dbc = MyCnf("~/.my.cnf").mysqldb_pars(sect)
        try:  
            conn = MySQLdb.connect( read_default_group=sect )   # huh, version variation in accepted params
        except MySQLdb.Error, e: 
            raise Exception("Error %d: %s " % ( e.args[0], e.args[1] ) )
        self.conn = conn
        self._size = None

    def execute_(self, cmd):
        cursor = self.conn.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute( cmd )
        return cursor

    def fetchall(self, cmd ):
        cursor = self.execute_(cmd)
        rows = cursor.fetchall()
        self.count = cursor.rowcount
        cursor.close()
        return rows

    def _query_size(self):
        sql = "select round(sum((data_length+index_length-data_free)/1024/1024),2) as TOT_MB from information_schema.tables where table_schema = '%(database)s' " % self.dbc
        return float(self(sql)[0]['TOT_MB'])
    def _get_size(self):
        if self._size is None:
             self._size = self._query_size()
        return self._size 
    size = property(_get_size, doc="Size estimate of the DB in MB ") 

    def __call__(self, cmd):
        log.debug(cmd)
        return self.fetchall(cmd)


class MyCnf(dict):
    def __init__(self, path = "~/.my.cnf"): 
        prime = {}
        cfp = ConfigParser(prime)
        paths = cfp.read( [os.path.expandvars(os.path.expanduser(p)) for p in path.split(":")] )
        self.cfp = cfp 
        self.sections = cfp.sections() 
        self.path = path
        self.paths  = paths
    def section(self, sect):
        return dict(self.cfp.items(sect))
    def mysqldb_pars(self, sect):
        pars = self.section(sect)
        want = "host user database password".split()
        return dict((k,pars[k]) for k in filter(lambda k:k in want,pars.keys()))


if __name__ == '__main__':
    db = DB("tmp_offline_db")
    print db.size


