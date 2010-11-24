"""
    Optimizing creation of numpy arrays from mysql queries 

    Summary findings (details in NOTES.txt )

       * MySQLdb 1.2.3 cursor usage leaky and slow (factor 10) , and cannot be push to big queries (1M rows)
       * 1.2.3 _mysql/fromiter is fast and convenient
       * cythoning can double the _mysql speed at cost of static numpy array types

"""
import sys, os

from timeit import Timer
import numpy as np
from env.npy.scan import Scan
        
import MySQLdb
import _mysql


from env.memcheck.mem import Ps
_ps = Ps(os.getpid())
rss = lambda:_ps.rss_megabytes


import env.mysql_np.dcspmthv as dcspmthv 
from env.mysql_np.npy import DB


class Fetch(dict):
    _sql = "select %(cols)s from %(tab)s limit %(limit)s  "
    _qry = {
       'DcsPmtHv':{
             'descr':[('SEQNO', 'i4'), ('ROW_COUNTER', 'i4'), ('ladder', 'i4'),('col','i4'),('ring','i4'),('voltage','f4'),('pw','i4')],
               'tab':'DcsPmtHv',
             'limit':10000,
             'voltage':lambda _:"coalesce(%s,0.)" % _,
                  'pw':lambda _:"coalesce(%s,0)" % _,
                  } 
            }

    def _cols(self):
        """ special coalese handling for some columns with NULLs that cause issue at MySQLdb level  """
        cols = []
        for name in self.colnames:
            cols.append( name in self and self[name](name) or name )
        return cols

    cols = property(_cols) 
    colnames = property(lambda self:map(lambda _:_[0], self['descr'])) 
    dtype = property(lambda self:np.dtype(self['descr']))
    sql  = property(lambda self:self._sql % self )
    connargs = property(lambda self:dict(read_default_file="~/.my.cnf", read_default_group=self['read_default_group'])) 
    limit = property( lambda self:int(self['limit']) )

    def __repr__(self):
        return "%s(\"%s\",%s)" % ( self.__class__.__name__, self.name , ",".join(["%s=\"%s\"" % _ for _ in self.kwargs.items() ] ) ) 

    def __init__(self, name, **kwargs):
        assert name in self._qry
        self.name = name
        self.kwargs = kwargs 
        self.update(self._qry[name]) 
        self['cols'] = ",".join(self.cols)
        self.update(kwargs)


class Pure(Fetch):
    """
       Problems with MySQLdb 1.2.3 ...
         * NULL columns are causing "TypeError: Cannot convert None to Decimal"
           forcing use of coalesce with default zeros for voltage and pw columns

      This has monotonic memory ???
       cannot push either approach to big queries ... due to memory death 
           http://www.coactivate.org/projects/topp-engineering/lists/topp-engineering-discussion/archive/2009/01/1231971483836/forum_view
      Only accepts the charset for  MYSQL_VERSION_ID >= 50007


    """
    def __call__(self, **kwargs ):
        self.update(kwargs)

        conn = MySQLdb.connect( **dict( self.connargs )  )
        cursor = conn.cursor()
        cursor.execute( self.sql )          ## this includes the time-consuming get_result

        if 0:
            a = np.zeros( ( self.limit, ) , dtype=self.dtype )
            for i,row in enumerate(cursor):
                a[i] = tuple(row)
            if 'verbose' in kwargs:print a
            a = None

        a = np.fromiter( (tuple(row) for row in cursor), dtype=self.dtype, count = self.limit  )
        if 'verbose' in kwargs:print a
        a = None

        cursor.close()    ## NOT CLOSING THE CURSOR LEAKS MORTALLY ... EVEN WITH IT ARE STILL LEAKIN LIKE SIEVE 
        conn.close()      ## LITTLE MEMORY IMPACT

class Xure(Fetch):
    """
          Avoid the leaky cursor  
          ... very healthy speed up only 2-3* slower than Cyth

    """
    def __call__(self, **kwargs ):
        self.update(kwargs)

        conn = _mysql.connect( **self.connargs )
        conn.query( self.sql )
        result = conn.get_result()      ## this takes the time
 
        a = np.fromiter( result , dtype=self.dtype, count = self.limit )
        
        result.clear()     ## ESSENTIAL MEMORY CLEANUP
        result = None 
        conn.close()

        if 'verbose' in kwargs:print a
        a = None


class Viadb(Fetch):
    """
         Exactly the same mem/time characteristics as Xure 
         with nice API for interactive usage ... and 
         it works for any generic query 
         (array dtype is introspected from the result )
    """
    def __call__(self, **kwargs ):
        self.update(kwargs)
       
        db = DB( **self.connargs )
        a = db( self.sql )
        db.close()

        if 'verbose' in kwargs:print a
        a = None


class Cyth(Fetch):
    """
        Factor of 10 faster than Pure* ... and no leaking (unlike Pure)
        does the shovelling from mysql rows into numpy array in C
        method 1 is approx twice the speed as method 0, 
        with the same memory  ... removal of bounds checking/wraparound 

    """     
    def __call__(self, **kwargs ):
        self.update(kwargs)
        
        conn = _mysql.connect( **self.connargs )
        conn.query( self.sql )
        result = conn.get_result()   ## this takes the time
        
        a = np.zeros( self.limit  , self.dtype )
        meth = getattr( dcspmthv , "fetch_rows_into_array_%d" % self['method'] )
        meth( result, a )

        result.clear()      ## ESSENTIAL MEMORY CLEANUP
        result = None 
        conn.close()
        
        if 'verbose' in kwargs:print a
        a = None



class Task(dict):
    _name  = "%(class_)s_%(kls)s_%(method)s"
    _setup = "gc.enable() ; from __main__ import %(kls)s ; obj = %(kls)s(\"%(name)s\", read_default_group=\"%(dbconf)s\" )  "
    _stmt = "obj(method=%(method)s, limit=%(limit)s) ; del obj ; obj=None " 

    setup = property( lambda self:self._setup % self )
    stmt =  property( lambda self:self._stmt % self )
    name =  property( lambda self:self._name % self )

    def __call__(self):
        timer = Timer( self.stmt, self.setup )
        try:
            time_=timer.timeit(1)  
        except:
            time_=-1
            timer.print_exc()
            sys.exit(1)

        res = dict( time_=time_, rss_=rss(), task_=self.name )
        self.update( **res ) 
        if 'verbose' in self:
            print self
        return res


class LimitScan(Scan):
    steps = 5 
    max   = 1000000   

class DebugScan(Scan):
    steps = 10 
    max = 10000



def check_technique(tech, kwargs):
    tech(**kwargs)

def test_techniques():
    tab, grp = "DcsPmtHv","client"
    kwargs = dict( method=1, verbose=1, limit=60000 )

    yield  check_technique, Pure( tab ,read_default_group=grp ), kwargs
    yield  check_technique, Xure( tab ,read_default_group=grp ), kwargs
    yield  check_technique, Cyth( tab ,read_default_group=grp ), kwargs
    yield  check_technique, Viadb( tab ,read_default_group=grp ), kwargs


def test_fromiter():
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" )   # _mysql.connection  instance 
    q = Fetch("DcsPmtHv", limit=100 )
    conn.query( q.sql )
    result = conn.get_result()      ## _mysql.result object    (slow)
    a = np.fromiter( result , dtype=q.dtype, count = q.limit )
    result.clear()     ## ESSENTIAL MEMORY CLEANUP
    result = None 
    conn.close()
    print repr(a)
    n = len(a)
    assert n == q.limit , ( n , q.limit ) 

def test_transitional():
    conn = MySQLdb.connect(  read_default_file="~/.my.cnf", read_default_group="client" )
    cursor = conn.cursor()
    q = Fetch("DcsPmtHv", limit=100 )

    # transitioning from cursors to lower level
    #
    # cursor.execute( q )   
    # does 
    #   _clear 
    #   _query  ... 
    #         _flush 
    #         cursor._result = Result(cursor) 
    #      BUT instantiation of the Result object 
    #         ... pulls the rows and flushes the result
    #             setting result to None
    #      SO NO GOOD    

    cursor._clear()
    # adapt from  from cursor._query 
    db = cursor._get_db()    ## the  _mysql.connection  instance 
    cursor._flush()          ## clears result 
    cursor._executed = q.sql
    db.query(q.sql)

    # adapt from  Result.__init__
    result = db.get_result(cursor.use_result)   ## _mysql.result object
    a = np.fromiter( result  , dtype=q.dtype )

    # after usage 
    result.clear()
    result = None
    print repr(a)

    n = len(a)
    assert n == q.limit , ( n , q.limit ) 




if __name__=="__main__":
    pass
    
    #for chk, tech, kwargs in test_techniques():
    #    chk(tech,kwargs)
    #test_fromiter()
    #test_transitional()


if 1: 
    base = dict(name="DcsPmtHv", dbconf="client", verbose=1 , limit="*" , method=0 )
    scargs = (
        #dict( kls="Pure", symbol="ro" ),
         dict( kls="Viadb", symbol="ro" ),
         dict( kls="Xure", symbol="bo" ),
         dict( kls="Cyth",  symbol="g^" , method=1 ),
    )
    for n, scarg in enumerate(scargs):
        if len(sys.argv) > 1 and int(sys.argv[1]) != n:continue  
        scan = LimitScan( **dict( base, **scarg) )
        print "starting scan %s " % repr(scan)

        for ctx in scan:
            tsk = Task(ctx) 
            res = tsk()
            scan( **res )   ## record result in the scan structure at this cursor point 

        print repr(scan.scan) 
        scan.save()

