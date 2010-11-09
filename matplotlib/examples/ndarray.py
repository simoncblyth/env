"""
    Bringing MySQLdb DBI query results into numpy/matplotlib realm 
   
        http://mail.scipy.org/pipermail/numpy-discussion/2010-September/052599.html


"""
import MySQLdb  
import numpy as np
import decimal, types
from datetime import datetime


class DB(object):
    def __init__(self, **kwa ):
        conn = MySQLdb.connect( **kwa )
        self.cursor = conn.cursor()

    def __call__(self, sql ):
        self.cursor.execute(sql)

    def _dtype(self):
        """
        Return an appropriate descriptor for a numpy array in which the
        result can be stored, by introspection of the cursor description.

        http://crucible.broadinstitute.org/browse/~raw,r=10032/CellProfiler/trunk/CPAnalyst/src/DBConnect.py

        """
        descr = []
        for (name, type_code, display_size, 
               internal_size, precision, scale, null_ok), flags in zip(self.cursor.description, 
                                          self.cursor.description_flags):
            conversion = self.cursor.connection.converter[type_code]
            if isinstance(conversion, list):
                fun2 = None
                for mask, fun in conversion:
                    fun2 = fun
                    if mask & flags:
                        break
            else:
                fun2 = conversion
            if fun2 in [decimal.Decimal, types.FloatType]:
                dtype = 'f8'
            elif fun2 in [types.IntType, types.LongType]:
                dtype = 'i4'
            elif fun2 in [types.StringType]:
                dtype = '|S%d'%(internal_size,)
            descr.append((name, dtype))
        return descr
    dtype = property( _dtype , doc=_dtype.__doc__ )


    def _iter(self):
        return (tuple (row) for row in self.cursor)
    iter = property( _iter )

    def _ndarray(self):
        """   
            An efficient? way of translating MySQLdb SQL query results 
            into numpy ndarrays ( actually structured/record arrays )

                 http://docs.scipy.org/doc/numpy/user/whatisnumpy.html 
                 http://docs.scipy.org/doc/numpy/user/basics.rec.html#structured-arrays
            
            Usage :
                from env.offline.dbconf import DBConf
                db = DB(**DBConf("prior").mysqldb )
                db("select p.SEQNO, ... )
                a = db.ndarray
                
                print a['SEQNO']

            For simple plotting use matplotlib functionality from ipython 
 
                 ipython -pylab
                 ipython -pylab ndarray.py 

            Provides field access my name and record access by number...
            eg a[0] a[-1]
 

                 plot( a['pedestalLow'], a['INSERTDATE'] )
                 clf()

                 hist( a['pedestalLow'] , bins=100 )
                 clf()
               

            NB imperative to definendarray size ahead of time for performance, 
                otherwise much copying will get done as new fixed length ndarrays are created  

        """
        return np.fromiter( self.iter, dtype=self.dtype, count = self.cursor.rowcount)
    ndarray = property( _ndarray , doc=_ndarray.__doc__ )
    
    def dump(self):
        for _ in self.iter:
            print _



class DT(object):
    """ 
         Usage :
              DT.vld("2010-01-01","%Y-%m-%d")

         provide dict that can be used to replace the Vld table DATETIME 
         columns, such as   
               %(TIMESTART)s etc.. 
         with the SQL strings that will turn the column specification into a decimal 
         number of days from a start date    

         Needed as numpy datetime handling not yet usable (?)
    """
    def vld(cls, **kwa):
        dt = cls(**kwa)
        return dict( (k,dt(k)) for k in "TIMESTART TIMEEND VERSIONDATE INSERTDATE".split() )
    vld = classmethod(vld)
    def __init__(self, dt0="2010-01-01" , fmt="%Y-%m-%d" ):
         self.ptn = "TIME_TO_SEC(timediff(v.%s,'" + str(datetime.strptime(dt0, fmt)) + "'))/(60*60*24) as %s "
    def __call__(self, col):
         return self.ptn % ( col, col ) 
         


if __name__ == '__main__':
    from env.offline.dbconf import DBConf
    db = DB(**DBConf("prior").mysqldb )

    tab = "CalibFeeSpec"
    limit = 100000
    db("select p.SEQNO,p.ROW_COUNTER, p.pedestalLow, %(TIMESTART)s, %(TIMEEND)s,%(VERSIONDATE)s,%(INSERTDATE)s from %(tab)s as p inner join %(tab)sVld as v on p.SEQNO = v.SEQNO limit %(limit)s " % dict(DT.vld(), tab=tab,limit=limit) )

    a = db.ndarray      
    print a['pedestalLow']
    print a['SEQNO']






