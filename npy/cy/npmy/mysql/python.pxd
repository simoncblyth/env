"""
   Attempt to follow the wrapping pattern of
 
        site-packages/Cython/Include/numpy.pxd


        http://docs.cython.org/src/userguide/extension_types.html

"""

import mysql.api as mysql
cimport mysql.api as mysql

cdef extern from "mysqlmod.h":
    ctypedef class mysql.python.connection [object _mysql_ConnectionObject]:
        cdef mysql.MYSQL connection
        cdef int open

    ctypedef class mysql.python.result [object _mysql_ResultObject]:
        cdef object conn
        cdef mysql.MYSQL_RES* result
        cdef int nfields
        cdef int use
        cdef object fields

    ctypedef class mysql.python.field [object _mysql_FieldObject]:
        cdef object result
        cdef mysql.MYSQL_FIELD field
        cdef unsigned int index


                                                   




