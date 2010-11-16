"""
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
import timeit
import numpy as np
import npmy
    
name = "DcsPmtHv"

class Fetch(dict):
    _sql = "select %(cols)s from %(tab)s limit %(limit)s  "
    _qry = {
       'DcsPmtHv':{
             'descr':[('SEQNO', 'i4'), ('ROW_COUNTER', 'i4'), ('ladder', 'i4'),('col','i4'),('ring','i4'),('voltage','f4'),('pw','i4')],
               'tab':'DcsPmtHv',
             'limit':10000,
                  } 
            }

    cols = property(lambda self:map(lambda _:_[0], self['descr'])) 
    dtype = property(lambda self:np.dtype(self['descr']))
    sql  = property(lambda self:self._sql % self )
    scanpath = classmethod(lambda cls,name:"ts_%s.npz" % name )
    scan = classmethod(lambda cls,name:np.load( cls.scanpath(name)) )
 
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
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        import MySQLdb
        self.conn = MySQLdb.connect(  read_default_file="~/.my.cnf", read_default_group=self['read_default_group'] )

    def __call__(self, **kwargs ):
        self.update(kwargs)
        self.cursor = self.conn.cursor()
        self.cursor.execute( self.sql )
        a = np.fromiter( (tuple(row) for row in self.cursor), dtype=self.dtype, count = self.cursor.rowcount)
        if 'verbose' in kwargs:print a

class Cyth(Fetch):
    def __init__(self, *args, **kwargs ):
        super(self.__class__, self).__init__(*args, **kwargs)
        import _mysql
        self.conn = _mysql.connect( read_default_file="~/.my.cnf", read_default_group=self['read_default_group'] )

    def __call__(self, **kwargs ):
        self.update(kwargs)
        conn = self.conn
        conn.query( self.sql )
        result = conn.get_result()
        a = np.zeros( int(self['limit']) , self.dtype )
        meth = getattr( npmy , "fetch_rows_into_array_%d" % self['method'] )
        meth( result, a )
        if 'verbose' in kwargs:print a




def test_cyth():
    fetch = Cyth( name ,read_default_group="client",limit="10")
    print fetch
    print fetch.sql
    print fetch.dtype
    fetch(method=1,verbose=1)

def test_pure():
    fetch = Pure( name ,read_default_group="client",limit="10")
    print fetch
    print fetch.sql
    print fetch.dtype
    fetch(method=1,verbose=1)


if __name__=="__main__":

    if 0:
        test_pure()
        test_cyth()
        import sys
        sys.exit(0)


    #lmax = 730000  ## causes a kill ... memory bug ?
    #lmax = 500000 + 1    
    lmax = 10000 + 1    

    klss = ("Pure", "Cyth") 
    methods = (1,)
    limits = np.arange(0,lmax,10000)

    ts = np.zeros( len(klss)*len(limits)*len(methods) , np.dtype([('kls','S10'),('method','i4'),('limit','i4'),('time','f4')]))
    #ts = np.zeros( (len(klss),len(limits),len(methods)) , np.float32 )
    
    n = 0
    for kls in klss:
        for method in methods:
            for limit in limits:
                timer = timeit.Timer("fetch(method=%d, limit=%d)" % (method, limit) , "from __main__ import %s ; fetch = %s(\"%s\", read_default_group=\"client\" )  " % ( kls, kls, name) )
                try:
                    t = timer.timeit(1)
                    ts[n] = (kls , method, limit, t )
                    print n, ts[n]    
                    n += 1
                except:
                    timer.print_exc()
                pass
            pass

    print ts
    np.savez( Fetch.scanpath(name) , ts=ts )  


