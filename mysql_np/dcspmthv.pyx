
cimport cython

import numpy as np
cimport numpy as np

cimport _mysql

cimport c_api

from libc.stdlib cimport const_char, strtof, atoi, atof


# dynamic typing sacrificed on performance altar 
cdef packed struct dtype_t:
    np.int32_t SEQNO, ROW_COUNTER
    np.int32_t ladder,col,ring
    np.float32_t voltage
    np.int32_t pw



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
    cdef c_api.MYSQL_ROW row 
    cdef c_api.my_ulonglong num_rows 
    cdef c_api.my_ulonglong max_rows 
    cdef unsigned int num_fields
    cdef c_api.MYSQL_RES* res 

    res        = result.result   
    num_rows   = c_api.mysql_num_rows( res )
    num_fields = c_api.mysql_num_fields( res )
    print "fetch_rows_into_array_1 : num_rows %d num_fields %s " % ( num_rows , num_fields ) 

    if a.shape[0] < num_rows:
        print "warning truncating to fit into array "
        max_rows = a.shape[0]
    else:
        max_rows = num_rows 


    for i in range(max_rows):
        row        = c_api.mysql_fetch_row( res )
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
