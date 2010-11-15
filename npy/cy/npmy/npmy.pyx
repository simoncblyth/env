"""

   fetch_row converts c strings from mysql into python strings ...
   which is a bit pointless as i am going to stuff into the numpy array 
   as floats/ints...

   but its a lot less effort to 
   numpy impinge above the fetch_row level ...
   as then i dont need to touch the bare MYSQL objects 

"""

import numpy as np
cimport numpy as np

import _mysql
cimport _mysql 

#from libc.stdlib cimport const_char, strtof
# fetch_rows has already converted to python strings 

# dynamic typing sacrificed on performance altar 
cdef packed struct qdt:
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

def fetch_rows_into_array(_mysql.result res, np.ndarray[qdt] qa):
    """
        fixed types for speed ... when dealing with huge queries 
        if need flexibility do some code generation 
    """ 
    cdef Py_ssize_t n, i  
    print "fetch_rows_into_array"
    print qa 
    n = res.num_rows()
    print n 
    for i in range(n):
        row = res.fetch_row()
        qa[i].SEQNO = int(row[0])
        qa[i].ROW_COUNTER = int(row[1])
        qa[i].ladder = int(row[2])
        qa[i].col = int(row[3])
        qa[i].ring = int(row[4])
        qa[i].voltage = float(row[5])
        qa[i].pw = int(row[6])
