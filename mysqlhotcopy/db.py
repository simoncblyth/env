#!/usr/bin/env python
"""

"""
import os, logging
from ConfigParser import ConfigParser

# allows non MySQL-python nodes to autodoc
try:
    import MySQLdb
except ImportError:
    MySQLdb = None


log = logging.getLogger(__name__)

class DB(object):
    def __init__(self, sect, database=None , group_concat_max_len=8192, group_by="SEQNO"):
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
        self("set @@group_concat_max_len = %(group_concat_max_len)s" % locals())
        self.group_by = group_by

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

    def fields( self, dbtab, skipfield):
        """
        :param dbtab: dbname and table specifcation string eg `channelquality_db.DqChannel` on same server as self
        :return: list of fields names

        ::

            mysql> select column_name from information_schema.columns where concat(table_schema,'.',table_name) = 'channelquality_db.DqChannel' ;
            +-------------+
            | column_name |
            +-------------+
            | SEQNO       | 
            | ROW_COUNTER | 
            | RUNNO       | 
            | FILENO      | 
            | CHANNELID   | 
            | OCCUPANCY   | 
            | DADCMEAN    | 
            | DADCRMS     | 
            | HVMEAN      | 
            | HVRMS       | 
            +-------------+
            10 rows in set (0.01 sec)

        """
        sql = "select column_name from information_schema.columns where concat(table_schema,'.',table_name) = '%(dbtab)s' " % locals()
        return filter(lambda _:not _ in skipfield, map(lambda _:_['column_name'], self(sql) ))

    def group_by_digest(self, dbtab, limit="0,10" ):
        """
        """
        group_by = self.group_by 
        fields = ",".join(self.fields(dbtab, skipfield=group_by.split()))
        sql = "select %(group_by)s,md5(group_concat(md5(concat_ws(',',%(fields)s)) separator ',')) as digest from %(dbtab)s group by %(group_by)s limit %(limit)s " % locals()
        return dict((d[group_by],d['digest']) for d in self(sql))

    def compare_by_digest(self, a, b,  limit="0,10" ):
        a = self.group_by_digest( a, limit ) 
        b = self.group_by_digest( b, limit ) 
        same = a == b 
        if not same:
            log.warn("compare_by_digest difference %s %s %s %s  " % (a,b,limit,group_by) )
            assert len(a) == len(b) , "length mismatch "
            assert a.keys() == b.keys() , "keys mismatch "
            for k in sorted(a.keys()):
                if a[k] != b[k]:
                    log.warn(" %s %s %s " % ( k, a[k], b[k] ))
            pass
        return same

    def group_by_range(self, dbtab):
        group_by = self.group_by
        return self("select min(%(group_by)s) as min, max(%(group_by)s) as max from %(dbtab)s " % locals())[0]

    def digest_table_scan(self, a, b, chunk=1000 ):
        ar = self.group_by_range(a)
        br = self.group_by_range(b)

        log.info("a %-30s %s " % (a, ar )) 
        log.info("b %-30s %s " % (b, br )) 

        if ar == br:
            num = ar['max'] - ar['min'] + 1 
        else:
            max = min(ar['max'], br['max'])
            assert ar['min'] == br['min']
            num = max - ar['min'] + 1 
           
        log.info("common max %s num %s " % (max, num ))
        offset = 0
        while offset < num:
            limit = "%(offset)s, %(chunk)s " % locals()
            same = self.compare_by_digest(a, b, limit )
            log.info( " %s : %s " % ( limit, same ) )
            offset += chunk


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

    def _get_datadir(self):
        return self("select @@datadir as datadir")[0]['datadir']
    datadir = property(_get_datadir, doc="Query DB server to find the datadir, eg /var/lib/mysql/ OR /data/mysql/ ")
        
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
    print "datadir:", srv.datadir
    for database in srv.databases:
        db = DB("mysqlhotcopy", database=database)
        print "%-40s %7s " % ( database,  db.size)




