"""

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

try:
    import npmy
except ImportError:
    print "you need to \"make\" to create the python/cython extension first "     
    sys.exit(1)


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
    scanpath = classmethod(lambda cls,name:"ts_%s.npz" % name )
    scan = classmethod(lambda cls,name:np.load( cls.scanpath(name)) )
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

    def __del__(self):
        if self.conn:
            self.close()

    def close(self):
        print "closing connection %s " % repr(self)
        self.conn.close()
        self.conn = None


class Pure(Fetch):
    """
       Problems with MySQLdb 1.2.3 ...
         * NULL columns are causing "TypeError: Cannot convert None to Decimal"
           forcing use of coalesce with default zeros for voltage and pw columns
         * cursor.rowcount giving -1 

    """
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    def __call__(self, **kwargs ):
        self.update(kwargs)
        conn = MySQLdb.connect( **self.connargs  )
        cursor = conn.cursor()
        cursor.execute( self.sql )

        if self['method'] == 0:

            a = np.ndarray( ( int(self['limit']), ) , dtype=self.dtype )
            for i,row in enumerate(cursor):
                a[i] = tuple(row)
            if 'verbose' in kwargs:print a
            a = None

        elif self['method'] == 1:
            a = np.fromiter( (tuple(row) for row in cursor), dtype=self.dtype, count = int(self['limit']) )
            if 'verbose' in kwargs:print a
            a = None

        conn.close()


class Cyth(Fetch):
    def __init__(self, *args, **kwargs ):
        super(self.__class__, self).__init__(*args, **kwargs)

    def __call__(self, **kwargs ):
        self.update(kwargs)
        
        conn = _mysql.connect( **self.connargs )
        conn.query( self.sql )
        result = conn.get_result()
        
        a = np.zeros( int(self['limit']) , self.dtype )
        meth = getattr( npmy , "fetch_rows_into_array_%d" % self['method'] )
        meth( result, a )
        if 'verbose' in kwargs:print a
        a = None

        conn.close()


def test_cyth():
    fetch = Cyth( "DcsPmtHv" ,read_default_group="client")
    print fetch
    fetch(method=1,verbose=1,limit=60000)

def test_pure():
    fetch = Pure( "DcsPmtHv" ,read_default_group="client")
    print fetch
    fetch(method=0,verbose=1,limit=60000)




class LimitScan(Scan):
    _name  = "%(class_)s_%(kls)s_%(method)s"
    _setup = "gc.enable() ; from __main__ import %(kls)s ; obj = %(kls)s(\"%(name)s\", read_default_group=\"%(dbconf)s\" )  "
    _stmt = "obj(method=%(method)s, limit=%(limit)s) ; del obj ; obj=None " 
    steps = 11 
    max   = 100000   ## 1000000 Pure is dying well before get to 1M 

    def __init__(self, *args, **kwargs ): 
        Scan.__init__(self, *args, **kwargs)
        scan = np.zeros( (self.steps,) , np.dtype([('limit','i4'),('time_','f4'),('rss_','f4')]) ) 
        scan['limit'] = np.linspace( 0, self.max , len(scan) )
        self.scan = scan

    def __call__(self):
        if 'verbose' in self:print self
        timer = Timer( self.stmt, self.setup )
        try:
            time_=timer.timeit(1)  
        except:
            time_=-1
            timer.print_exc()
            sys.exit(1)
        pass
        Scan.__call__(self, time_=time_, rss_=rss() )

    def close(self):
        self.scan = None   

 
class DebugScan(LimitScan):
    steps = 10 
    max = 10000




if __name__=="__main__":

    if 0:
        test_pure()
        #test_cyth()
        import sys
        sys.exit(0)


    name, dbconf = "DcsPmtHv", "client"
    scargs = (
               dict( kls="Pure", method=1, limit="*" , symbol="ro" , verbose=1 , name=name, dbconf=dbconf ),
               dict( kls="Cyth", method=0, limit="*" , symbol="go" , verbose=1 , name=name, dbconf=dbconf ),
               dict( kls="Cyth", method=1, limit="*" , symbol="g^" , verbose=1 , name=name, dbconf=dbconf ),
             )

    for n, scarg in enumerate(scargs):
        if len(sys.argv) > 1 and int(sys.argv[1]) != n:continue  
        scan = LimitScan( **scarg )
        print "starting scan %s " % repr(scan)
        scan.run()
        print repr(scan.scan) 
        scan.save()
        scan.close()
        scan = None

