#!/usr/bin/env python
"""
simtab.py
===========


Quick and dirty sqlite, for cases when using SQLAlchemy or django is overkill

http://www.sqlite.org/lang_conflict.html
http://www.sqlite.org/lang_insert.html

Operation with python2.3
--------------------------

In order to work correctly with py2.3 which does not include sqlite3 as
standard it is necessary to install the `pysqlite2` module allowing::

    from pysqlite2 import dbapi2 as sqlite

Install that with yum via (you might need to enable EPEL repository to find it)::

   sudo yum install python-sqlite2


py2.3 where EPEL mirror doesnt provide dependencies
----------------------------------------------------

Grab RPMs::

    curl -L -O http://download.fedoraproject.org/pub/epel/4/i386/python-sqlite2-2.3.3-4.el4.i386.rpm
    curl -L -O http://download.fedoraproject.org/pub/epel/4/i386/sqlite-3.3.6-0.3.el4.i386.rpm
    curl -L -O http://download.fedoraproject.org/pub/epel/4/i386/sqlite-devel-3.3.6-0.3.el4.i386.rpm

Check their content::

    rpm -qlp python-sqlite2-2.3.3-4.el4.i386.rpm
    rpm -qlp sqlite-devel-3.3.6-0.3.el4.i386.rpm
    rpm -qlp sqlite-3.3.6-0.3.el4.i386.rpm

And install::

    rpm -i python-sqlite2-2.3.3-4.el4.i386.rpm
    rpm -i sqlite-devel-3.3.6-0.3.el4.i386.rpm
    rpm -i sqlite-3.3.6-0.3.el4.i386.rpm


"""
import os, logging
log = logging.getLogger(__name__)

try:
    import sqlite3 as sqlite
except ImportError:
    try:
        from pysqlite2 import dbapi2 as sqlite
    except ImportError:
        import sqlite

def version_info():
    _ = sqlite 
    print (_.__file__,_.sqlite_version,_.sqlite_version_info,_.version,_.version_info) 


class Table(list):
    """
    Interact with sqlite3 tables, append dicts to this list then insert them 
    """
    bigtable = False
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
         self.conf = dict(tn=tn, path=pathv)

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


        Old SQLite dont like "CREATE TABLE IF NOT EXISTS"


sqlite> SELECT sql FROM sqlite_master WHERE type='table' AND name='oomon';
sql                                    
---------------------------------------
CREATE TABLE oomon (date text,val real)


        """

        pk = kwa.pop('pk',None)   # optionally use comma delimited 
            
        fields = kwa.keys()
        types = kwa.values()

        if pk:
            pkf = ["PRIMARY KEY( %s )" % pk] 
        else:    
            pkf = []

        fields_sql = ",".join(["%s %s" % (k, v) for k,v in zip(fields,types)] + pkf )

        create_sql = "CREATE TABLE %(tn)s (%(fields_sql)s)" % locals()
        check_schema_q = self.cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='%(tn)s'" % locals() )
        if check_schema_q is None:
            log.debug("check_schema_q is None" )
            check_schema = []
        else:    
            check_schema = check_schema_q.fetchall()

        log.debug("check_schema : %s " % str(check_schema)) 
        if len(check_schema) == 0:
            log.debug("_create: %s " % create_sql )
            self.cursor.execute(create_sql)
        else:
            check_schema = check_schema[0][0]
            assert check_schema == create_sql , ("schema mismatch", check_schema, create_sql )
            log.debug("table %(tn)s exists already and has expected schema :" % locals() + check_schema  )
        pass
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
         if not tn is None:
             ti = self.cursor.execute("pragma table_info(%s)" %  tn )
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
         replace them.  But this is minting new hidden `rowid`, so try `insert or update`

         Note that the dicts in the list can contain more key:value pairs than needed without harm
         """
         qxn = self.qxn
         tn = self.tn
         entries = [] 
         for d in self:
             def value(k):
                 return d.get(k,'NULL')
             vals = list(map( value, self.fields )) 
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


    def all(self, sql):
        """
        """ 
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def getone(self, sql):
        lret = map(lambda _:_[0],self(sql))
        assert len(lret) == 1, (sql,lret)
        return lret[0]

    def asdict(self, kf, vf, sql=None ):
        """
        :param kf: function that returns key from the dict of query columns 
        :param vf: function that returns value from the dict of query columns 
        :param sql: query, default of None corresponds to `select * from %(tn)s`
        """
        if not sql:
            sql = "select * from %(tn)s "
        d = {}
        for r in self(sql % self.conf, fdict=True):
            d[kf(r)] = vf(r)
        return d

    def iterdict(self, sql, labels=None):
        """
        :param sql: sql to perform

        Caution sql needs to be of general form ``select * from whatever``
        """ 
        if labels:
            labels = labels.split(",") 
        else:
            labels = self.fields
        for row in self.cursor.execute(sql):
            yield dict(zip(labels,row))

    def listdict(self, sql, labels=None):
        """
        :param sql: sql to perform
        :param labels: comma delimited ordered list of column labels used for dict keys 
        :return: list of dicts

        Caution field/label list must correspond to the columns returned by the 
        query unless the query is of the form ``select * from whatever`` in which 
        case the full list of fields is used.
        """ 
        if labels:
            labels = labels.split(",") 
        else:
            labels = self.fields
        l = []
        for row in self.cursor.execute(sql):
            l.append(dict(zip(labels,row)))
        return l    

    def dump(self, sql="select rowid, * from %(tn)s ;" ):
        ctx = dict(self.conf, sql=sql % self.conf )
        q = os.popen("echo '%(sql)s' | sqlite3 %(path)s " % ctx).read()
        print(q)  


    @classmethod
    def FromLines(cls, txtpath, priority_ = lambda line:0):
        """
        :param txtpath: eg /tmp/somelist.txt
        :param priority_: function with line argument that returns a priority integer
        :return tab: Table instance

        Creates database at eg /tmp/somelist.txt.db containing a single table named "somelist". 
        With fields:

        idx
            0-based line number in the file
        line
            from the file
        priority
            int returned from priority_(line) function 

        """
        txtpath = os.path.abspath(txtpath)
        basename = os.path.basename(txtpath)
        stem, ext = os.path.splitext(basename)
        tablename = stem 
        dbpath = "%s.db" % txtpath
        tab = cls(dbpath, tablename, idx="int primary key", line="text", priority="int")
        lines = map(str.strip,open(txtpath, "r").readlines())
        for idx,line in enumerate(lines):
            tab.add(idx=idx, line=line, priority=priority_(line))
        pass
        tab.insert()
        log.info("creating tablename %s in db %s " % (tablename, dbpath))
        return tab 



class BigTable(Table):
    bigtable = True



def demo():
    t = Table("demo.db", "tgzs", date="text", size="real" )
    t.add( date='2001-10-10', size=10 )
    t.add( date='2002-10-10', size=20 )
    print(t)
    t.insert()


def test_iterdict():
    dbp = "/tmp/env/simtab/tscm_backup_check.db"
    #t = Table(dbp, "tgzs", nodepath="text primary key", node="text", dir="text", date="text", size="real" ) 
    t = Table(dbp, "tgzs") 
    print(t.fields)
    for d in t.iterdict("select * from tgzs"):
        print(d) 

def test_small_table_id():
    urls = "a b c d e f g".split()
    rt = Table("/tmp/repos.db", "repos" , id="integer", url="text primary key"  )
    for url in urls:
        rd = rt.asdict(kf="url", vf="id" )              # convenient approach for small tables
        rid = rd.get(url, max(rd.values() or [0])+1 )   # when url already stored get the id from that otherwise increment the max id by one
        rt.add( id=rid, url=url, _insert=True )  
    pass     
    rt.dump()


if __name__ == '__main__':
    level=logging.DEBUG
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    hdlr = logging.StreamHandler()
    formatter = logging.Formatter(logformat)
    hdlr.setFormatter(formatter)
    log.addHandler(hdlr)
    log.setLevel(level)

    version_info()

    ct = Table("/tmp/colors.db", "colors", label="text primary key" )
    ct("CREATE UNIQUE INDEX idx_colors ON colors (label);") 
    colors = "red green blue red green blue".split()
    for col in colors:
        ct.add( label=col , _insert=True)
        print("lastrowid %d" % ct.cursor.lastrowid) 
    pass
    ct.dump()




