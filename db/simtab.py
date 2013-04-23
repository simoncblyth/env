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
         Open or create SQLite DB at `path`, if `tn` and `kwa` are provided create the 
         table with fields specified by the `kwa` if the table does not exist already.

         If you want to change a table schema, make sure to drop it first with interactive `sqlite3`

         :param path: to sqlite3 DB file 
         :param tn: table name
         :param kwa: key value pairs defining field names and types
         """ 
         list.__init__(self) 
         conf = dict(tn=tn, path=path)
         ## also kwa begging with _ are taken into conf which is used for sql interpolation
         for k,v in kwa.items():
             if k[0] == '_':
                 conf[k[1:]] = kwa.pop(k)

         pathv = os.path.expanduser(os.path.expandvars(path))
         dirv = os.path.dirname(pathv)
         if not os.path.isdir(dirv):
             log.info("creating directory %s " % dirv )         
             os.makedirs(dirv)         
         log.debug("opening DB path %s resolves to %s dir %s " % (path,pathv,dirv) ) 
         conn = sqlite.connect(pathv)
         conn.text_factory = str    # rather than unicode default, see http://stackoverflow.com/questions/3425320/sqlite3-programmingerror-you-must-not-use-8-bit-bytestrings-unless-you-use-a-te
         cursor = conn.cursor()
         self.path = path
         self.conn = conn 
         self.cursor = cursor 
         self.tn = tn
         self.conf = conf

         if kwa:
             self._create(tn, kwa)
         else:
             self._tableinfo(tn)   
             # pragma table_info not working for py2.3 (or maybe old sqlite) so must always spell out the kwa in that case

    def add(self, **kwa):
        _insert = kwa.pop('_insert',False)
        self.append(kwa)
        if _insert:
            self.insert(clear=True)

    def _create(self, tn, kwa):
        """
        Creates table `tn` if one of that name does not already exists with
        fieldnames and types as specified by the `kwa`.

        For single column PK can specify the pk along with the type ie::

              id="int primary key"

        For compound PK specify the primary key columns separately with::

              pk="col1, col2"

        This will be used to construct SQL of structure::

              CREATE TABLE something (col1, col2, col3, PRIMARY KEY (col1, col2)); 

        :param tn: table name
        :param kwa: `dict` of field names and types  
        """

        pk = kwa.pop('pk',None)   # optionally use comma delimited 
            
        fields = kwa.keys()
        types = kwa.values()

        if pk:
            pkf = ["PRIMARY KEY( %s )" % pk] 
        else:    
            pkf = []

        fields_sql = ",".join(["%s %s" % (k, v) for k,v in zip(fields,types)] + pkf )
        create_sql = "CREATE TABLE IF NOT EXISTS %(tn)s (%(fields_sql)s)" % locals()
        self.cursor.execute(create_sql)
        log.debug("_create: %s " % create_sql )

        self.qxn = ",".join(["?" for _ in range(len(kwa.keys()))] )
        self.fields = kwa.keys()


  
    def _tableinfo(self, tn):
         """
         Introspect information about an existing table, allowing subsequent 
         access to a table to do so just by name.

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

    def insert(self, clear=False):
         """
         :param clear: when `True` clear the collected list of dicts after insertion

         Uses ``insert or replace`` so new entries with the same PK as existing ones will 
         replace them.  

         Note that the dicts in the list can contain more key:value pairs than needed without harm
         """
         qxn = self.qxn
         tn = self.tn
         entries = [] 
         for d in self:
             def value(k):
                 return d.get(k,'NULL')
             vals = map( value, self.fields ) 
             entries.append(  vals )
         sql = 'INSERT OR REPLACE INTO %(tn)s VALUES (%(qxn)s)' % locals()
         log.debug("sql %s " % sql )     
         log.debug("\n".join(map(str,entries)) )     
         self.cursor.executemany(sql, entries )
         self.conn.commit()
         if clear:
             self[:]=[]
       

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
            pass
        pass    

    def asdict(self, kf, vf, sql=None ):
        """
        :param kf: function with single query dict argument that returns the desired key 
        :param vf: function with single query dict argument that returns the desired key 
        :param sql:
        """
        if not sql:
            sql = "select * from %(tn)s "
        d = {}
        for r in self(sql % self.conf, fdict=True):
            d[kf(r)] = vf(r)
        return d

    def iterdict(self, sql, fields=None):
        """
        :param sql: sql to perform

        Caution sql needs to be of general form ``select * from whatever``
        """ 
        if fields:
            fields = fields.split(",") 
        else:
            fields = self.fields
        for row in self.cursor.execute(sql):
            yield dict(zip(fields,row))

    def listdict(self, sql, labels=None):
        """
        :param sql: sql to perform

        Caution sql needs to be of general form ``select * from whatever``
        """ 
        if labels:
            labels = labels.split(",") 
        else:
            labels = self.fields
        l = []
        for row in self.cursor.execute(sql):
            l.append(dict(zip(labels,row)))
        return l    

    def dump(self, sql="select * from %(tn)s ;" ):
        ctx = dict(self.conf, sql=sql % self.conf )
        print self.conf
        print os.popen("echo '%(sql)s' | sqlite3 %(path)s " % ctx).read()

def demo():
    t = Table("demo.db", "tgzs", date="text", size="real" )
    t.add( date='2001-10-10', size=10 )
    t.add( date='2002-10-10', size=20 )
    print t
    t.insert()


def test_iterdict():
    dbp = "/tmp/env/simtab/tscm_backup_check.db"
    #t = Table(dbp, "tgzs", nodepath="text primary key", node="text", dir="text", date="text", size="real" ) 
    t = Table(dbp, "tgzs") 
    print t.fields
    for d in t.iterdict("select * from tgzs"):
        print d 

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    urls = "a b c d e f g".split()

    rt = Table("/tmp/repos.db", "repos" , id="integer", url="text primary key"  )
    for url in urls:
        rd = rt.asdict(kf="url", vf="id" )              # convenient approach for small tables
        rid = rd.get(url, max(rd.values() or [0])+1 )   # when url already stored get the id from that otherwise increment the max id by one
        rt.add( id=rid, url=url, _insert=True )  
    pass     
    rt.dump()




