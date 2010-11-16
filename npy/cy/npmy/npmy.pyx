"""

   fetch_row converts c strings from mysql into python strings ...
   which is a bit pointless as i am going to stuff into the numpy array 
   as floats/ints...

   but its a lot less effort to 
   numpy impinge above the fetch_row level ...
   as then i dont need to touch the bare MYSQL objects 

"""

import cython

import numpy as np
cimport numpy as np

import _mysql
cimport _mysql 

cimport mysql

from libc.stdlib cimport const_char, strtof, atoi, atof

# dynamic typing sacrificed on performance altar 
cdef packed struct dtype_t:
    np.int32_t SEQNO, ROW_COUNTER
    np.int32_t ladder,col,ring
    np.float32_t voltage
    np.int32_t pw




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

def fetch_rows_into_array_0(_mysql.result result, np.ndarray[dtype_t, ndim=1] a not None):
    """
        fixed types for speed ... when dealing with huge queries 
        if need flexibility do some code generation 
    """ 
    cdef Py_ssize_t n, i  
    print "fetch_rows_into_array"
    n = result.num_rows()
    for i in range(n):
        row = result.fetch_row()
        a[i].SEQNO = int(row[0])
        a[i].ROW_COUNTER = int(row[1])
        a[i].ladder = int(row[2])
        a[i].col = int(row[3])
        a[i].ring = int(row[4])
        a[i].voltage = float(row[5])
        a[i].pw = int(row[6])



@cython.boundscheck(False)
@cython.wraparound(False)
def fetch_rows_into_array_1(_mysql.result result, np.ndarray[dtype_t, ndim=1] a not None):
    """
        fixed types for speed ... when dealing with huge queries 
        if need flexibility do some code generation 
    """ 
    cdef Py_ssize_t i, j   
    cdef mysql.MYSQL_ROW row 
    cdef mysql.my_ulonglong num_rows 
    cdef unsigned int num_fields
    cdef mysql.MYSQL_RES* res 

    res        = result.result   
    num_rows   = mysql.mysql_num_rows( res )
    num_fields = mysql.mysql_num_fields( res )
    print "fetch_rows_into_array_1 : num_rows %d num_fields %s " % ( num_rows , num_fields ) 

    for i in range(num_rows):
        row        = mysql.mysql_fetch_row( res )
        a[i].SEQNO = atoi(row[0])
        a[i].ROW_COUNTER = atoi(row[1])
        a[i].ladder = atoi(row[2])
        a[i].col = atoi(row[3])
        a[i].ring = atoi(row[4])
        a[i].voltage = atof(row[5])
        a[i].pw = atoi(row[6])

    return num_rows
