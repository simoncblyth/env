
== View MySQL queries as numpy record arrays ==

{{{
n [1]: from env.mysql_np import DB

In [2]: db = DB()

In [3]:  a = db("show tables")
   ...: 

In [4]: a
Out[4]: 
array([('CalibFeeSpec',), ('CalibFeeSpecVld',), ('CalibPmtSpec',),
       ('CalibPmtSpecVld',), ('DaqCalibRunInfo',), ('DaqCalibRunInfoVld',),
       ('DaqRawDataFileInfo',), ('DaqRawDataFileInfoVld',),
       ('DaqRunConfig',), ('DaqRunInfo',), ('DaqRunInfoVld',),
       ('DcsAdTemp',), ('DcsAdTempVld',), ('DcsPmtHv',), ('DcsPmtHvVld',),
       ('FeeCableMap',), ('FeeCableMapVld',), ('LOCALSEQNO',),
       ('SimPmtSpec',), ('SimPmtSpecVld',)], 
      dtype=[('Tables_in_offline_db_20101111', '|S64')])

In [5]: b = db("select * from CalibFeeSpec")

In [6]: b
Out[6]: 
array([(3, 1, 2, 4, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0),
       (3, 2, 1, 2, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
       (4, 1, 536937985, 0, 0.0, 0.0, 62.044511610414098, 2.1510729036762699,
0.0, 0.0),
       ...,
       (113, 206, 536940814, 0, 0.0, 0.0, 59.017800000000001, 1.49282, 0.0,
0.0),
       (113, 207, 536940815, 0, 0.0, 0.0, 54.418799999999997,
1.5282199999999999, 0.0, 0.0),
       (113, 208, 536940816, 0, 0.0, 0.0, 66.911600000000007, 1.56138, 0.0,
0.0)], 
      dtype=[('SEQNO', '<i4'), ('ROW_COUNTER', '<i4'), ('channelId', '<i4'),
('status', '<i4'), ('pedestalHigh', '<f8'), ('sigmaPedestalHigh', '<f8'),
('pedestalLow', '<f8'), ('sigmaPedestalLow', '<f8'), ('thresholdHigh', '<f8'),
('thresholdLow', '<f8')])

In [7]: 
}}}



== issues ==

  * NULL handling 
  * date handling
  * type conversion correspondence checks 


=== mysql-python patch issues ===

  * fetch_ndarrayfast 
      * failing for empty OR limit 0  query 
      * datetime wrong on big endian PPC ???


== development history ==

  * started with mysql-python ~1.3.0 (unreleased) + numpy 1.5
     * due to better source layout that makes cythoning easier

  * upped to numpy ~2.0.0 (unreleased) for datetime support

  * discover that django interface to mysql-python does not support 1.3.0
    * so move back to last release 1.2.3
  
  * features of mysql-pyhon 1.2.3 
    * no headers : makes difficult to cython
    * _mysql.result is not iterable (unlike 1.3.0)    
       * added iterability in patch 
    


== implementation notes ==


       4 levels ..

           0)   api.pxd : C API/library supplied by MYSQL
                   wrapper to allow cython to understand the base MYSQL types


           1)  _mysql.pxd : _mysql CPython module (from MySQL-python adustman)
                   wrapper for access to _mysql low level MySQL-python functionality   (not the higher level MySQLdb yet ... for sanity)
                   connect/query/get_result

           2)  npy.pyx  Cython extension module

                   Cython extension module
                      ... aiming to add numpy array result fetching to this 
                     

           3) test_npmysql.py    python usage of the extension module




== C implementation of array creation ==

  * dtype fields contains the byte offsets...


{{{
In [7]: for k,v in a.dtype.fields.items():print k,v
   ...: 
VERSIONDATE (dtype('datetime64[us]'), 34)
TASK (dtype('int32'), 26)
SEQNO (dtype('int32'), 0)
TIMEEND (dtype('datetime64[us]'), 12)
SITEMASK (dtype('int8'), 20)
TIMESTART (dtype('datetime64[us]'), 4)
AGGREGATENO (dtype('int32'), 30)
SIMMASK (dtype('int8'), 21)
INSERTDATE (dtype('datetime64[us]'), 42)
SUBSITE (dtype('int32'), 22)

In [8]: dt = a.dtype

In [9]: dt.kind
Out[9]: 'V'

In [10]: dt.str
Out[10]: '|V50'

In [11]: dt.num
Out[11]: 22

In [12]: dt.itemsize
Out[12]: 50


In [14]: for n in a.dtype.names:print "%-15s %s " % ( n, a.dtype.fields[n] )
   ....: 
SEQNO           (dtype('int32'), 0) 
TIMESTART       (dtype('datetime64[us]'), 4) 
TIMEEND         (dtype('datetime64[us]'), 12) 
SITEMASK        (dtype('int8'), 20) 
SIMMASK         (dtype('int8'), 21) 
SUBSITE         (dtype('int32'), 22) 
TASK            (dtype('int32'), 26) 
AGGREGATENO     (dtype('int32'), 30) 
VERSIONDATE     (dtype('datetime64[us]'), 34) 
INSERTDATE      (dtype('datetime64[us]'), 42) 


}}}


