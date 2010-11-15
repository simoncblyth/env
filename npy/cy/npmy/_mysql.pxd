"""
   Attempt to follow the wrapping pattern of
 
        site-packages/Cython/Include/numpy.pxd


"""

from mysql cimport MYSQL, MYSQL_RES, MYSQL_FIELD

cdef extern from "mysqlmod.h":
    ctypedef class _mysql.connection [object _mysql_ConnectionObject]:
        cdef MYSQL connection
        cdef int open

    ctypedef class _mysql.result [object _mysql_ResultObject]:
        cdef object conn
        cdef MYSQL_RES* res
        cdef int nfields
        cdef int use
        cdef object fields

    ctypedef class _mysql.field [object _mysql_FieldObject]:
        cdef object result
        cdef MYSQL_FIELD field
        cdef unsigned int index


                                                   




