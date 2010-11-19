"""

   fetch_row converts c strings from mysql into python strings ...
   which is a bit pointless as i am going to stuff into the numpy array 
   as floats/ints...

   but its a lot less effort to 
   numpy impinge above the fetch_row level ...
   as then i dont need to touch the bare MYSQL objects 


  testing enum access 
     avoid spilling guts like this by implementing the 
     result description translation into numpy descr tuple 
     within mysql.???  
  

"""

cimport cython
import numpy as np

cimport numpy as np
cimport _mysql
cimport api as mysql

DECIMAL     = mysql.MYSQL_TYPE_DECIMAL
TINY        = mysql.MYSQL_TYPE_TINY
SHORT       = mysql.MYSQL_TYPE_SHORT
LONG        = mysql.MYSQL_TYPE_LONG
FLOAT       = mysql.MYSQL_TYPE_FLOAT
DOUBLE      = mysql.MYSQL_TYPE_DOUBLE
NULL_       = mysql.MYSQL_TYPE_NULL
TIMESTAMP   = mysql.MYSQL_TYPE_TIMESTAMP
LONGLONG    = mysql.MYSQL_TYPE_LONGLONG
INT24       = mysql.MYSQL_TYPE_INT24
DATE        = mysql.MYSQL_TYPE_DATE   
TIME        = mysql.MYSQL_TYPE_TIME
DATETIME    = mysql.MYSQL_TYPE_DATETIME
YEAR        = mysql.MYSQL_TYPE_YEAR
NEWDATE     = mysql.MYSQL_TYPE_NEWDATE
ENUM        = mysql.MYSQL_TYPE_ENUM
SET         = mysql.MYSQL_TYPE_SET
TINY_BLOB   = mysql.MYSQL_TYPE_TINY_BLOB
MEDIUM_BLOB = mysql.MYSQL_TYPE_MEDIUM_BLOB
LONG_BLOB   = mysql.MYSQL_TYPE_LONG_BLOB
BLOB        = mysql.MYSQL_TYPE_BLOB
VAR_STRING  = mysql.MYSQL_TYPE_VAR_STRING
STRING      = mysql.MYSQL_TYPE_STRING
GEOMETRY    = mysql.MYSQL_TYPE_GEOMETRY



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

def fetch_rows_into_array_0( result, np.ndarray[dtype_t, ndim=1] a not None ):
    """
        fixed types for speed ... when dealing with huge queries 
        if need flexibility do some code generation 
    """  
    cdef Py_ssize_t i, num_rows, max_rows  
    num_rows = result.num_rows()
    print "fetch_rows_into_array_0 %d " % num_rows
    if a.shape[0] < num_rows:
        print "warning truncating to fit into array "
        max_rows = a.shape[0]
    else:
        max_rows = num_rows 

    for i in range(max_rows):
        row = result.fetch_row()
        if row[0]: 
            a[i].SEQNO = int(row[0])
        if row[1]: 
            a[i].ROW_COUNTER = int(row[1])
        if row[2]: 
            a[i].ladder = int(row[2])
        if row[3]: 
            a[i].col = int(row[3])
        if row[4]: 
            a[i].ring = int(row[4])
        if row[5]: 
            a[i].voltage = float(row[5])
        if row[6]: 
            a[i].pw = int(row[6])



@cython.boundscheck(False)
@cython.wraparound(False)
def fetch_rows_into_array_1(_mysql.result result, np.ndarray[dtype_t, ndim=1] a not None):
    """
        fixed types for speed ... when dealing with huge queries 
        if need flexibility do some code generation 

        hmm to access struct members without hardcoding the names
        ... using offsets based on know types in the numpy struct
        (is the numpy record array element a normal struct ?)
        need to really hack it  eg  
            http://stackoverflow.com/questions/2043871/how-can-i-get-set-a-struct-member-by-offset
        
        BUT need code generation anyhow ... as this needs to be type fixed 
            so just do the struct access in code generation stage 

        THIS APPROACH ALSO SIMPLER


         numpy as np.nan ... but what happens in plotting with  that ???


    """ 
    cdef Py_ssize_t i, j   
    cdef mysql.MYSQL_ROW row 
    cdef mysql.my_ulonglong num_rows 
    cdef mysql.my_ulonglong max_rows 
    cdef unsigned int num_fields
    cdef mysql.MYSQL_RES* res 

    res        = result.result   
    num_rows   = mysql.mysql_num_rows( res )
    num_fields = mysql.mysql_num_fields( res )
    print "fetch_rows_into_array_1 : num_rows %d num_fields %s " % ( num_rows , num_fields ) 

    if a.shape[0] < num_rows:
        print "warning truncating to fit into array "
        max_rows = a.shape[0]
    else:
        max_rows = num_rows 


    for i in range(max_rows):
        row        = mysql.mysql_fetch_row( res )
        if row == NULL:
            continue

        if row[0] != NULL:
            a[i].SEQNO = atoi(row[0])
        if row[1] != NULL:
            a[i].ROW_COUNTER = atoi(row[1])
        if row[2] != NULL:
            a[i].ladder = atoi(row[2])
        if row[3] != NULL:
            a[i].col = atoi(row[3])
        if row[4] != NULL:
            a[i].ring = atoi(row[4])
        if row[5] != NULL:
            a[i].voltage = atof(row[5])
        if row[6] != NULL:
            a[i].pw = atoi(row[6])


    return num_rows
