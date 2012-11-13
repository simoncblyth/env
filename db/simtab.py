#!/usr/bin/env python
"""
Quick and dirty sqlite, for cases when using SQLAlchemy or django is overkill

http://www.sqlite.org/lang_conflict.html
http://www.sqlite.org/lang_insert.html

"""
import os, sqlite3, logging
log = logging.getLogger(__name__)

class Table(list):
    """
    Interact with sqlite3 tables, append dicts to this list then insert them 
    """
    def __init__(self, path, tn=None , **kwa ):
         list.__init__(self) 
         pathv = os.path.expanduser(os.path.expandvars(path))
         dirv = os.path.dirname(pathv)
         if not os.path.isdir(dirv):
             log.info("creating directory %s " % dirv )         
             os.makedirs(dirv)         
         log.info("opening DB path %s resolves to %s dir %s " % (path,pathv,dirv) ) 
         conn = sqlite3.connect(pathv)
         cursor = conn.cursor()
         self.path = path
         self.conn = conn 
         self.cursor = cursor 

         if kwa:
             self._create(tn, kwa)
         self._tableinfo(tn)

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
         for row in self.cursor.execute("pragma table_info(%s)" %  tn ):
             index,name,dtype,nonnull,default,primary = row
             fields.append(name)
             types.append(dtype)
             pass

         self.fields = fields
         self.types = types
         self.qxn = ",".join(["?" for _ in range(len(fields))] )
         self.tn = tn

    def __repr__(self):
         return "%s %s %s " % ( self.__class__.__name__, self.tn, repr(dict(zip(self.fields,self.types))) )


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
         log.debug("entries %s " % entries)     
         self.cursor.executemany('INSERT OR REPLACE INTO %(tn)s VALUES (%(qxn)s)' % locals(), entries )
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


def demo():
    t = Table("demo.db", "tgzs", date="text", size="real" )
    t.add( date='2001-10-10', size=10 )
    t.add( date='2002-10-10', size=20 )
    print t
    t.insert()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #t = Table("scm_backup_check.db", "tgzs", nodepath="text primary key", node="text", dir="text", date="text", size="real" )
    t = Table("scm_backup_check.db", "tgzs") 
    print t.fields
    for d in t("select * from tgzs", fdict=True):
        print d 

