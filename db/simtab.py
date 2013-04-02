#!/usr/bin/env python
"""
Quick and dirty sqlite, for cases when using SQLAlchemy or django is overkill

http://www.sqlite.org/lang_conflict.html
http://www.sqlite.org/lang_insert.html

Operation with python2.3
~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to work correctly with py2.3 which does not include sqlite3 as
standard it is necessary to install the `pysqlite2` module allowing::

    from pysqlite2 import dbapi2 as sqlite

Install that with yum via (you might need to enable EPEL repository to find it)::

   sudo yum install python-sqlite2


"""
import os, logging
log = logging.getLogger(__name__)

try:
    import sqlite3 as sqlite
except ImportError:
    from pysqlite2 import dbapi2 as sqlite


class Table(list):
    """
    Interact with sqlite3 tables, append dicts to this list then insert them 
    """
    def __init__(self, path, tn=None , **kwa ):
         """
         :param path: to sqlite3 DB file 
         :param tn: table name
         :param kwa: key value pairs defining field names and types
         """ 
         list.__init__(self) 
         pathv = os.path.expanduser(os.path.expandvars(path))
         dirv = os.path.dirname(pathv)
         if not os.path.isdir(dirv):
             log.info("creating directory %s " % dirv )         
             os.makedirs(dirv)         
         log.info("opening DB path %s resolves to %s dir %s " % (path,pathv,dirv) ) 
         conn = sqlite.connect(pathv)
         cursor = conn.cursor()
         self.path = path
         self.conn = conn 
         self.cursor = cursor 
         self.tn = tn

         if kwa:
             self._create(tn, kwa)
             self.qxn = ",".join(["?" for _ in range(len(kwa.keys()))] )
             self.fields = kwa.keys()
         else:
             self._tableinfo(tn)   
             # pragma table_info not working for py2.3 (or maybe old sqlite) so must always spell out the kwa in that case

    def add(self, **kwa):
        self.append(kwa)

    def _create(self, tn, kwa):
        fields = kwa.keys()
        types = kwa.values()
        fields_sql = ",".join(["%s %s" % (k, v) for k,v in zip(fields,types)])
        create_sql = "CREATE TABLE IF NOT EXISTS %(tn)s (%(fields_sql)s)" % locals()
        self.cursor.execute(create_sql)
        log.info("_create: %s " % create_sql )
  
    def _tableinfo(self, tn):
         """
         http://www.sqlite.org/pragma.html#pragma_table_info

         column name, data type, whether or not the column can be NULL, and the default value for the column
         """ 
         fields = []
         types = []
         ti = self.cursor.execute("pragma table_info(%s)" %  tn )
         print ti
         for row in ti:
             index,name,dtype,nonnull,default,primary = row
             fields.append(name)
             types.append(dtype)
             pass

         self.fields = fields
         self.types = types
         self.qxn = ",".join(["?" for _ in range(len(fields))] )

    def smry(self):
         return "%s %s %s " % ( self.__class__.__name__, self.tn, repr(dict(zip(self.fields,self.types))) )

    def __repr__(self):
         return "%s %s " % ( self.__class__.__name__, self.tn )


    def insert(self):
         """
         Uses ``insert or replace`` so new entries with the same PK as existing ones will 
         replace them.  
         """
         qxn = self.qxn
         tn = self.tn
         entries = [] 
         for d in self:
             vals = map( lambda k:d.get(k,'NULL'), self.fields ) 
             entries.append(  vals )
         log.info("entries %s " % entries)     
         sql = 'INSERT OR REPLACE INTO %(tn)s VALUES (%(qxn)s)' % locals()
         log.info("sql %s " % sql )     
         self.cursor.executemany(sql, entries )
         self.conn.commit()
        
    def __call__(self, sql , fdict=False ):
        """
        :param sql: sql to perform
        :param fdict: when the sql is of form ``select * from tgzs`` can use this to return field dicts, not usable for general queries
        """ 
        for row in self.cursor.execute(sql):
            if fdict: 
                yield dict(zip(self.fields,row))
            else:
                yield row
    def iterdict(self, sql):
        """
        :param sql: sql to perform

        Caution sql needs to be of general form ``select * from whatever``
        """ 
        for row in self.cursor.execute(sql):
            yield dict(zip(self.fields,row))



def demo():
    t = Table("demo.db", "tgzs", date="text", size="real" )
    t.add( date='2001-10-10', size=10 )
    t.add( date='2002-10-10', size=20 )
    print t
    t.insert()



if __name__ == '__main__':
    logging.basicConfig()

    dbp = "/tmp/env/simtab/tscm_backup_check.db"
    #t = Table(dbp, "tgzs", nodepath="text primary key", node="text", dir="text", date="text", size="real" ) 
    t = Table(dbp, "tgzs") 
    print t.fields
    for d in t.iterdict("select * from tgzs"):
        print d 

