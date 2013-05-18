#!/usr/bin/env python
"""

"""
import os, logging
import MySQLdb
from ConfigParser import ConfigParser
log = logging.getLogger(__name__)

class DB(object):
    def __init__(self, sect, database=None ):
        """
        :param sect: used as the `read_default_group` in MySQLdb connection 
        :param database: name of DB that overrides any setting within the 
        """
        self.sect = sect
        dbc = MyCnf("~/.my.cnf").mysqldb_pars(sect, database=database)
        self.dbc = dbc
        self.database = dbc.get('db', None)
        log.debug("connecting to %s " % dict(dbc, passwd="***"))
        try:  
            conn = MySQLdb.connect( **dbc )   # huh, version variation in accepted params
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
        sql = "select round(sum((data_length+index_length-data_free)/1024/1024),2) as TOT_MB from information_schema.tables where table_schema = '%(db)s' " % self.dbc
        return float(self(sql)[0]['TOT_MB'])
    def _get_size(self):
        if self._size is None:
             self._size = self._query_size()
        return self._size 
    size = property(_get_size, doc="Size estimate of the DB in MB ") 

    def _get_databases(self):
        """
        This query gives fewer results than `show databases`, which demands skips to avoid errors in getting sizes 
        #skip = "hello hello2 other test_noname tmp_cascade_2 tmp_dbitest tmp_tmp_offline_db_2".split()  
        """
        sql = "select distinct(table_schema) from information_schema.tables"
        return map(lambda _:_['table_schema'],self(sql))
    databases = property(_get_databases, doc="List of database names obtained from information_schema.tables") 
        
    def __call__(self, cmd):
        log.debug(cmd)
        return self.fetchall(cmd)


class MyCnf(dict):
    def __init__(self, path = "~/.my.cnf"): 
        prime = {}
        cfp = ConfigParser(prime)
        paths = cfp.read( [os.path.expandvars(os.path.expanduser(p)) for p in path.split(":")] )
        log.debug("MyCnf read %s " % repr(paths) )
        self.cfp = cfp 
        self.sections = cfp.sections() 
        self.path = path
        self.paths  = paths
    def section(self, sect):
        return dict(self.cfp.items(sect))
    def mysqldb_pars(self, sect, database=None):
        """
        Annoyingly mysql-python need these keys

        `host` host to connect
        `user` user to connect as
        `passwd` password to use
        `db` database to use

        whereas mysql uses slightly different ones

        `host`   
        `database` 
        `user`
        `password`

        Normally can avoid this annoyance using::

            conn = MySQLdb.connect( read_default_group=sect )   

        but when need to impinge `database/db` settings this is not possible.
        """
        my2mp = dict(host="host",user="user",password="passwd",database="db", socket="unix_socket")
        my = self.section(sect)
     
        mp = {}
        for k in filter(lambda k:k in my2mp,my.keys()):  # key translation, mysql to mysql-python
            mp[my2mp[k]] =  my[k]

        if database:
            mp["db"] = database 
        log.debug("translate mysql config %s into mysql-python config %s " % ( dict(my,password="***") , dict(mp,passwd="***") ))
        return mp 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    srv = DB("mysqlhotcopy")  
    for database in srv.databases:
        db = DB("mysqlhotcopy", database=database)
        print "%-40s %7s " % ( database,  db.size)




