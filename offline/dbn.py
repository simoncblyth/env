"""
    Bringing MySQLdb DBI query results into numpy/matplotlib realm 
   
        http://mail.scipy.org/pipermail/numpy-discussion/2010-September/052599.html

     Interesting "_getdatafromsql" using masks to avoid NULLs  ... also uses scikits.timeseries
        http://projects.scipy.org/scikits/browser/branches/pierregm/hydroclimpy/scikits/hydroclimpy/io/sqlite.py

    Timing different approaches to go from MySQL into numpy
        http://mail.scipy.org/pipermail/numpy-discussion/2006-November/024692.html

    Andy Dustman (MySQLdb author) on how to implement a direct MySQLdb to numpy bridge
        http://mail.python.org/pipermail/db-sig/2000-April/001251.html

    SQLite into numpy performance discussion , comparing with pytables 
        http://www.mail-archive.com/numpy-discussion@scipy.org/msg00014.html

    scikits.timeseries : for timeseries handling/plotting/etc... 
        http://pytseries.sourceforge.net/
        based on http://docs.scipy.org/doc/numpy/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray 
  
       http://pytseries.sourceforge.net/lib.database.html
           masked approach to NULL handling 

 
    pytables based on HDF ... claims to be a lot faster that SQL databases 
       http://www.pytables.org/moin/HowToUse

"""
import MySQLdb  
import decimal, types
from datetime import datetime

from env.offline.dbconf import DBConf


class DBn(object):
    def __init__(self, *args, **kwa ):
        
        tab = kwa.pop('tab',None)
        d = {}
        if len(args) == 1:
           d.update( **DBConf(args[0]).mysqldb )
        d.update( kwa )

        conn = MySQLdb.connect( **d )
        self.cursor = conn.cursor()
        self._tab = None
        if tab:
            self.tab = tab 
        pass

    def get_tab( self ):
        return self._tab 
    def set_tab( self, tab):
        """
             On assigning the active table lookup column names 
        """
        self._tab = tab
        self("describe %s" % tab)
        self.cols = map( lambda _:_[0] , self.iter ) 
        print "setting tab %s , describe gives columns %s " % ( tab , repr(self.cols) )
    tab = property( get_tab , set_tab ) 


    def __repr__(self):
        return "<DB %s %s %s>" % ( self.tab , repr(self.cols) , len(self) )


    def __call__(self, *args , **kwa  ):
        """
             Simple explicut execution from sql arguments or more 
             involved query interpolation from kwa        
        """
        if len(args) > 0:
            for arg in args:
                self.cursor.execute(arg)
        else:
            q = Qry( **dict( kwa , tab = self.tab, cols = self.cols ) ) 
            self.cursor.execute( repr(q) )
            print "query yields rowcount %s " % self.cursor.rowcount
        return self 


    def _dtype(self):
        """
        Return an appropriate descriptor for a numpy array in which the
        result can be stored, by introspection of the cursor description.

        http://crucible.broadinstitute.org/browse/~raw,r=10032/CellProfiler/trunk/CPAnalyst/src/DBConnect.py

        NB does not depend on numpy 

        TODO 
              ... modify this to work at a lower level 
              ... without leaky/slow cursors 


In [5]: result.describe()
Out[5]: 
(('SEQNO', 3, 1, 11, 11, 0, 0),
 ('ROW_COUNTER', 3, 3, 11, 11, 0, 0),
 ('ladder', 1, 1, 4, 4, 0, 1),
 ('col', 1, 1, 4, 4, 0, 1),
 ('ring', 1, 1, 4, 4, 0, 1),
 ('coalesce(voltage,0.)', 5, 7, 8, 8, 2, 1),
 ('coalesce(pw,0)', 8, 1, 4, 4, 0, 1))

In [6]: result
Out[6]: <_mysql.result object at b7947be4>


   In [3]: npmy.INTEGER
   Out[3]: 3

   In [4]: npmy.TINYINT
   Out[4]: 1

   In [8]: npmy.DOUBLE
   Out[8]: 5





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

    def __len__(self):
        return self.cursor.rowcount

    def _numpy(self):
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

             THIS IS CRAZY FROM PERFORMANCE POINT OF VIEW FOR LARGE QUERIES

                    _mysql result C ==> ( python tuple ) ==> numpy ndarray C 
  
        """
        try:
            import numpy as np
        except ImportError:
            print "missing numpy installation "
            return None 
        return np.fromiter( self.iter, dtype=self.dtype, count = self.cursor.rowcount)
    numpy = property( _numpy , doc=_numpy.__doc__ )
    
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
         

class Qry(dict):
    """
         parameterized select query with where-limit-offset tail   
    """
    dbi_sql = "select %(paycol)s, %(TIMESTART)s, %(TIMEEND)s from %(tab)s as p inner join %(tab)sVld as v on p.SEQNO = v.SEQNO where %(where)s limit %(limit)s offset %(offset)s " 
    vld_sql = "select v.SEQNO, %(TIMESTART)s, %(TIMEEND)s, %(VERSIONDATE)s, %(INSERTDATE)s from %(tab)s as v where %(where)s limit %(limit)s offset %(offset)s " 
    non_sql = "select %(paycol)s from %(tab)s as p where %(where)s limit %(limit)s offset %(offset)s " 

    defaults = {
              'base':dict( offset=0 , limit=100000 , where='1=1' ),
          'DcsPmtHv':dict( ladder=0, col=2, ring=3, where='ladder=%(ladder)s and col=%(col)s and ring=%(ring)s and voltage is not NULL and pw is not NULL ',)
                }
 
    def __init__(self, *arg, **kwa ):
        for t in ('base',kwa['tab']):
            self.update( self.defaults.get(t,{}) ) 
        self.update(kwa)
        self['paycol'] = ",".join(["p.%s" % col for col in self['cols'] ])
        self.update( DT.vld() )
        print dict(self)

    def __repr__(self):
        if self['tab'].upper().endswith('VLD'):
             sql = self.vld_sql 
        elif self['tab'] in "LOCALSEQNO GLOBALSEQNO DaqRunConfig".split():
             sql = self.non_sql 
        else:
             sql = self.dbi_sql  
        return sql % self % self 




if __name__ == '__main__':
    
    db = DB( "prior" )

    #db.tab = "DcsPmtHv"
    #a2 = db( ladder=0,col=2,ring=3 ).numpy   
    #a1 = db( ladder=0,col=1,ring=3 ).numpy  
    #  clf() ; plot( a1['TIMESTART'][300:], a1['voltage'][300:] , "b-")

    db.tab = "CalibFeeSpec"
    a3 = db().numpy      

    db.tab = "CalibFeeSpecVld"
    a4 = db().numpy




