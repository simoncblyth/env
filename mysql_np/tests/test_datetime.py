import _mysql
import numpy as np

conn = None

def setup():
    global conn
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" )   # _mysql.connection  instance 

def teardown():
    global conn
    conn.close()

def test_version():
    v = np.version.version
    assert v.startswith('2.0.0') , "numpy datetime handling only available from 2.0.0 ? "

def test_create():
     a = np.array( (1,) , dtype=np.datetime64 )
     a[0] = "2009-12-09 10:43:33"
     assert  a[0].__class__.__name__  == "datetime64" 
     #a[0] = None   fails must set to datetime or date object

def test_vld_fromiter():
    descr = [ 
          ('SEQNO',"i8",),
          ('TIMESTART',"datetime64[us]",),       
          ('TIMEEND',"datetime64[us]",),      
          ('SITEMASK',"i8",), 
          ('SIMMASK',"i8",),
          ('SUBSITE',"i8",),
          ('TASK',"i8",), 
          ('AGGREGATENO',"i8",), 
          ('VERSIONDATE',"datetime64[us]",),        
          ('INSERTDATE',"datetime64[us]",),         
       ]

    dt = np.dtype(descr) 
    conn.query( "select * from CalibFeeSpecVld limit 10" )       ## a table without NULL dates
    result = conn.get_result()   
    a = np.fromiter( result , dtype=dt ) 
    print a 

    for field in "TIMESTART TIMEEND VERSIONDATE INSERTDATE".split():
        k = a[field].dtype.kind
        assert k == 'M' , k 


if __name__ == '__main__':
    setup()

    test_version()
    test_create() 
    test_vld_fromiter()

    teardown()


