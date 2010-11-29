"""
    mysql-python 1.3.0 (is unreleased .. and onhold ..)
       * implements the row iterator in C , *10 speedup 
       * not compatible with django (as unreleased)

    mysql-python 1.2.3 
       * real slow numpy array construction 



"""
from npy import numpy_descr
import numpy as np
import MySQLdb

class SlowDB(object):
    def __init__(self, *args, **kwargs ):
        if args:
            kwargs.update( read_default_group=args[0] )         # some sugar 
        kwargs.setdefault( "read_default_file", "~/.my.cnf" ) 
        kwargs.setdefault( "read_default_group",  "client" )   # _mysql.connection  instance 
        conn = MySQLdb.connect( **kwargs )
        cursor = conn.cursor()
        
        self.conn = conn
        self.cursor = cursor
        self.q = None

    def __call__(self, q ):
        self.q = q 
        cursor = self.cursor
        
        cursor.execute( str(q) )
        count = cursor.rowcount
        descr = numpy_descr( cursor.description )
        dtype = np.dtype( descr ) 

        #print count, descr, dtype
        a = np.fromiter( (tuple(row) for row in cursor), dtype=dtype , count = count  )
        return a

    def close(self):
        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None



if __name__=='__main__':
    pass
    s = SlowDB("client")
    a = s("show tables")
    print repr(a)

    #s.close()





