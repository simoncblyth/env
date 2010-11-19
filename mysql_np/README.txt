
== View MySQL DB Tables as numpy record arrays ==

{{{
In [1]: from env.mysql_np import DBnp

In [2]: db = DBnp()

In [3]: a = db("show tables")
dtd [('Tables_in_offline_db_20101111', '|S64')] 
dtd [('Tables_in_offline_db_20101111', '|S64')] 
fetch_np dtype([('Tables_in_offline_db_20101111', '|S64')])  using count 20 

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

fetch_np dtype([('SEQNO', '<i4'), ('ROW_COUNTER', '<i4'), ('channelId',
'<i4'), ('status', '<i4'), ('pedestalHigh', '<f8'), ('sigmaPedestalHigh',
'<f8'), ('pedestalLow', '<f8'), ('sigmaPedestalLow', '<f8'), ('thresholdHigh',
'<f8'), ('thresholdLow', '<f8')])  using count 1984 

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

In [7]: len(b)
Out[7]: 1984

}}} 


== issues ==

  * NULL handling 





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





