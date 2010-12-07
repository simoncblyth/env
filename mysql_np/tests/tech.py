import sys, inspect
import numpy as np
import MySQLdb
import _mysql 

try:
    import env.mysql_np.dcspmthv as dcspmthv 
except ImportError:
    pass


from env.mysql_np.npy import DB


__all__ = ['Pure', 'Xure','Viadb', 'Cyth', ] 


class Tech(dict):
    __repr__ = lambda self:self.__class__.__name__
    def _members(cls):
        # all tech.Tech subclasses from module "tech" 
        return inspect.getmembers( sys.modules[cls.__module__] , lambda _:inspect.isclass(_) and issubclass(_, cls ) and _ is not cls and _.active )
    members   = classmethod( _members )    
    names     = classmethod( lambda cls:map(lambda _:_[0]  , cls.members() ))
    classes   = classmethod( lambda cls:map(lambda _:_[1]  , cls.members() ))
    callables = classmethod( lambda cls:map(lambda _:_[1](), cls.members() ))



class Pure(Tech):
    """
       Problems with MySQLdb 1.2.3 ...
         * NULL columns are causing "TypeError: Cannot convert None to Decimal"
           forcing use of coalesce with default zeros for voltage and pw columns

      This has monotonic memory ???
       cannot push either approach to big queries ... due to memory death 
           http://www.coactivate.org/projects/topp-engineering/lists/topp-engineering-discussion/archive/2009/01/1231971483836/forum_view
      Only accepts the charset for  MYSQL_VERSION_ID >= 50007


    """
    symbol = "ro"
    active = True 
    def __call__(self, q,  **kwargs ):
        self.update(kwargs)

        conn = MySQLdb.connect( **dict( q.connargs )  )
        cursor = conn.cursor()
        cursor.execute( str(q) )          ## this includes the time-consuming get_result

        if 0:
            a = np.zeros( ( q.limit, ) , dtype=np.dtype(q.descr) )
            for i,row in enumerate(cursor):
                a[i] = tuple(row)
            if 'verbose' in kwargs:print a
            a = None

        a = np.fromiter( (tuple(row) for row in cursor), dtype=np.dtype(q.descr), count = int(q.limit)  )
        if 'verbose' in kwargs:print a
        a = None

        cursor.close()    ## NOT CLOSING THE CURSOR LEAKS MORTALLY ... EVEN WITH IT ARE STILL LEAKIN LIKE SIEVE 
        conn.close()      ## LITTLE MEMORY IMPACT

class Xure(Tech):
    """
          Avoid the leaky cursor  
          ... very healthy speed up only 2-3* slower than Cyth

    """
    symbol = "go"
    active = False
    def __call__(self, q, **kwargs ):
        self.update(kwargs)

        conn = _mysql.connect( **q.connargs )
        conn.query( str(q) )
        result = conn.get_result()      ## this takes the time
 
        a = np.fromiter( result , dtype=np.dtype(q.descr), count = q.limit )
        
        result.clear()     ## ESSENTIAL MEMORY CLEANUP
        result = None 
        conn.close()

        if 'verbose' in kwargs:print a
        a = None


class Viadb(Tech):
    """
         Exactly the same mem/time characteristics as Xure 
         with nice API for interactive usage ... and 
         it works for any generic query 
         (array dtype is introspected from the result )
    """
    symbol = "bo"
    active = True
    def __call__(self, q, **kwargs ):
        self.update(kwargs)
       
        db = DB( **q.connargs )
        a = db( str(q) )
        db.close()

        if 'verbose' in kwargs:print a
        a = None


class Cyth(Tech):
    """
        Factor of 10 faster than Pure* ... and no leaking (unlike Pure)
        does the shovelling from mysql rows into numpy array in C
        method 1 is approx twice the speed as method 0, 
        with the same memory  ... removal of bounds checking/wraparound 

    """     
    symbol = "r^"
    active = False
    def __call__(self, q, **kwargs ):
        self.update(kwargs)
        
        conn = _mysql.connect( **q.connargs )
        conn.query( str(q) )
        result = conn.get_result()   ## this takes the time
        
        a = np.zeros( q.limit  , np.dtype(q.descr) )
        meth = getattr( dcspmthv , "fetch_rows_into_array_%d" % q.method )
        meth( result, a )

        result.clear()      ## ESSENTIAL MEMORY CLEANUP
        result = None 
        conn.close()
        
        if 'verbose' in kwargs:print a
        a = None



class Mynp(Tech):
    symbol = "ro"
    active = True 
    def __call__( self, q , **kwargs ):
        self.update(kwargs)
        
        conn = _mysql.connect( **q.connargs )
        conn.query( str(q) )
        result = conn.store_result()   ## this takes the time
        a = result.fetch_nparray()
        conn.close()
        
        if 'verbose' in kwargs:print repr(a)
        a = None

class Mynpfast(Tech):
    symbol = "go"
    active = True 
    def __call__( self, q , **kwargs ):
        self.update(kwargs)
        
        conn = _mysql.connect( **q.connargs )
        conn.query( str(q) )
        result = conn.store_result()   ## this takes the time

        #dtype = result.npdescr()
        #print repr(dtype)

        a = result.fetch_nparrayfast()
        conn.close()
        
        if 'verbose' in kwargs:print repr(a)
        a = None



if __name__ == '__main__':
    pass

