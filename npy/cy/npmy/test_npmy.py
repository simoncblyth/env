"""

   MEMORY LEAK CHECKING WITH 

  In [3]: MySQLdb.version_info
  Out[3]: (1, 3, 0, 'final', 0)

[blyth@cms01 MySQLdb-2.0]$ hg tip
changeset:   83:e705129ff06f
branch:      MySQLdb
tag:         tip
parent:      82:ffe9e5ca17e0
parent:      80:6ec608cdd19c
user:        Andy Dustman <adustman@users.sourceforge.net>
date:        Tue Aug 31 22:28:13 2010 -0400
summary:     Merge some Kyle stuff




   PROGESSIVE COMMENTING / DE-COMMENTING

       * skip array loading from the query result
       * skip array creation      
          * little change in memory/timings !!!
       * skip doing the query
          * memory/timings drastically reduced and flat
       * put back array creation 
          * timings equal-ish and memory flat when using np.zeros (ndarray is a bit quicker)

       * put back the query and get_result 
          * back to monotonic memory ==> MySQLdb/_mysql or my useage of it is culprit

       * for MySQLdb 1.3.0  
            * find that MUST cursor.close() TO AVOID LEAK
            * connection.close() IS NOT ENOUGH  



   ISSUES ...

    * monotonic memory ... when run multiple scans in the same process




    * tinyints could be used for ladder/col/ring ... but leave at i4 initially   
    * voltage and pw go null quite a lot ... avoid for now by "not null" exclusion  

    * defaul mysql is not to '''use''' 
        store_result  : results stored in client
        use_result    : results stay on server, until fetched ..... for HUGE resultsets
        result = conn.store_result() old 1.2.2 API is not longer there in 1.3
         NB num_rows will not be valid for "use_result" until all rows are fetched

   OOPS ... MEMORY DEATH ?

   fetch_rows_into_array_1 : num_rows 730000 num_fields 7 
(730000, 7.9681329727172852)
Killed


    Grab the timescan array with ...

       from test_npmy import Fetch
       npz = Fetch.scan("DcsPmtHv")  
       ts = npz['ts']

    For plotting the timescan use :

       ipython ts_plt.py 


    For interactive plot development 

       ipython ts_plt.py -pylab


"""
import sys, os
from timeit import Timer
import numpy as np
from env.npy.scan import Scan
        
import MySQLdb
import _mysql

import gc
from env.memcheck.mem import Ps
_ps = Ps(os.getpid())
rss = lambda:_ps.rss_megabytes

import mysql.npy as npy

#try:
#    import npmy
#except ImportError:
#    print "you need to \"make\" to create the python/cython extension first "     
#    sys.exit(1)


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

    """
    def __call__(self, **kwargs ):
        self.update(kwargs)

        conn = MySQLdb.connect( **self.connargs  )
        cursor = conn.cursor()
        cursor.execute( self.sql )          ## this includes the time-consuming get_result

        if 0:
            a = np.zeros( ( int(self['limit']), ) , dtype=self.dtype )
            for i,row in enumerate(cursor):
                a[i] = tuple(row)
            if 'verbose' in kwargs:print a
            a = None

        a = np.fromiter( (tuple(row) for row in cursor), dtype=self.dtype, count = int(self['limit']) )
        if 'verbose' in kwargs:print a
        a = None

        cursor.close()    ## NOT CLOSING THE CURSOR LEAKS MEMORY TERRIBLY 
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
 
        a = np.fromiter( result , dtype=self.dtype, count = int(self['limit']) )
        
        result.clear()     ## ESSENTIAL MEMORY CLEANUP
        result = None 
        conn.close()

        if 'verbose' in kwargs:print a
        a = None


class Cyth(Fetch):
    """
        Factor of 10 faster than Pure* but leaks like a sieve
        does the shovelling from mysql rows into numpy array in C

        method 1 is approx twice the speed as method 0, 
        with the same memory  ... removal of bounds checking/wraparound 

    """     
    def __call__(self, **kwargs ):
        self.update(kwargs)
        
        conn = _mysql.connect( **self.connargs )
        conn.query( self.sql )
        result = conn.get_result()   ## this takes the time
        
        a = np.zeros( int(self['limit']) , self.dtype )
        meth = getattr( npy , "fetch_rows_into_array_%d" % self['method'] )
        meth( result, a )

        result.clear()      ## ESSENTIAL MEMORY CLEANUP
        result = None 
        conn.close()
        
        if 'verbose' in kwargs:print a
        a = None



def test_cyth():
    fetch = Cyth( "DcsPmtHv" ,read_default_group="client")
    print fetch
    fetch(method=1,verbose=1,limit=60000)

def test_pure():
    fetch = Pure( "DcsPmtHv" ,read_default_group="client")
    print fetch
    fetch(method=0,verbose=1,limit=60000)



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



if __name__=="__main__":
    pass


if 0:
    """
         avoiding leaky cursors, and much faster 
         but can i still use the "description" for convenience of
         dynamic data definition for whatever query as done in DBn
 
    """
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" )   # _mysql.connection  instance 
    q = Fetch("DcsPmtHv", limit=100 )
    conn.query( q.sql )
    result = conn.get_result()      ## _mysql.result object    (slow)

    a = np.fromiter( result , dtype=q.dtype, count = int(q['limit']) )

    #result.clear()     ## ESSENTIAL MEMORY CLEANUP
    #result = None 
    #conn.close()
    print a

if 0:
    """
           transitioning from cursors to lower level 
    """
    conn = MySQLdb.connect(  read_default_file="~/.my.cnf", read_default_group="client" )
    cursor = conn.cursor()

    q = Fetch("DcsPmtHv", limit=100 )

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
    print a


if 0:
    test_pure()

if 0:
    test_cyth()

if 1: 
    base = dict(name="DcsPmtHv", dbconf="client", verbose=1 , limit="*" , method=0 )
    scargs = (
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

