"""
   Attempt to follow the wrapping pattern of
 
        site-packages/Cython/Include/numpy.pxd


"""

cimport mysql

cdef extern from "mysqlmod.h":
    ctypedef class _mysql.connection [object _mysql_ConnectionObject]:
        cdef mysql.MYSQL connection
        cdef int open

    ctypedef class _mysql.result [object _mysql_ResultObject]:
        cdef object conn
        cdef mysql.MYSQL_RES* result
        cdef int nfields
        cdef int use
        cdef object fields

    ctypedef class _mysql.field [object _mysql_FieldObject]:
        cdef object result
        cdef mysql.MYSQL_FIELD field
        cdef unsigned int index


                                                   




