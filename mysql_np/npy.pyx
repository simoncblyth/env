"""
 

"""

import numpy as np
cimport numpy as np
import  _mysql
cimport _mysql
cimport c_api 

## guesses ...
numpy_dt = {
   c_api.MYSQL_TYPE_DECIMAL     : "f8" ,
   c_api.MYSQL_TYPE_TINY        : "i1" ,
   c_api.MYSQL_TYPE_SHORT       : "i2" ,
   c_api.MYSQL_TYPE_LONG        : "i4" ,
   c_api.MYSQL_TYPE_FLOAT       : "f4" ,
   c_api.MYSQL_TYPE_DOUBLE      : "f8" ,
   c_api.MYSQL_TYPE_NULL        : ""   ,
   c_api.MYSQL_TYPE_TIMESTAMP   : "|S10" , 
   c_api.MYSQL_TYPE_LONGLONG    : "i8"   , 
   c_api.MYSQL_TYPE_INT24       : "i8"   , 
   c_api.MYSQL_TYPE_DATE        : "|S10" , 
   c_api.MYSQL_TYPE_TIME        : "|S10" ,
   c_api.MYSQL_TYPE_DATETIME    : "|S10" ,
   c_api.MYSQL_TYPE_YEAR        : "|S10" ,
   c_api.MYSQL_TYPE_NEWDATE     : "|S10" ,
   c_api.MYSQL_TYPE_ENUM        : "i4"   ,
   c_api.MYSQL_TYPE_SET         : "i4"   ,
   c_api.MYSQL_TYPE_TINY_BLOB   : "" ,
   c_api.MYSQL_TYPE_MEDIUM_BLOB : "" ,
   c_api.MYSQL_TYPE_LONG_BLOB   : "" ,
   c_api.MYSQL_TYPE_BLOB        : "" ,
   c_api.MYSQL_TYPE_VAR_STRING  : "|S%d" ,
   c_api.MYSQL_TYPE_STRING      : "|S%d" ,
   c_api.MYSQL_TYPE_GEOMETRY    : "" ,
}


class DB(object):
    def __init__(self, **kwargs ):
        kwargs.setdefault( "read_default_file", "~/.my.cnf" ) 
        kwargs.setdefault( "read_default_group",  "client" )   # _mysql.connection  instance 
        conn = _mysql.connection( **kwargs )
        self.conn = conn
        self.q = None

    def __call__(self, q ):
        self.q = q 
        self.conn.query( q )
        return fetch(self.conn) 

    def close(self):
        self.conn.close()


def fetch(_mysql.connection conn):

    result = conn.get_result()
    count = conn.affected_rows()
    describe = result.describe()

    descr = []   
    for (name, type_code, display_size, internal_size, precision, scale, null_ok) in describe:
        dt = numpy_dt.get(type_code,"f8")
        if '%' in dt:
            dt = dt % internal_size
        descr.append( (name, dt ) )
    dtype = np.dtype(descr)

    a = np.fromiter(result, dtype=dtype, count=count )
    
    result.clear()   ## ESSENTIAL MEMORY CLEAN UP 
    result = None
    return a 




def look(_mysql.result res):
    print "look"
    print res
    for f in res.describe():
        print f
    #print res.result   Cannot convert 'MYSQL_RES *' to Python object
    print res.nfields
    print res.use
    for f in res.fields:
        print f 
        #print f.index attribute error
        print f.result
    #row = res.fetch_row()   
    #print row 

