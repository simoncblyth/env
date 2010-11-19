"""
   Hack affording cython access inside mysql-python structs
   The mysql-python C module name is _mysql which must correspond
   to the name of this.
   
   Tis fragile ...
      * renaming to somthing other than  _mysql.pxd  fails
      * attempts to place in pkg hierarchy also fails 

   Follows the wrapping pattern of
      site-packages/Cython/Include/numpy.pxd

      http://docs.cython.org/src/userguide/extension_types.html
"""

cimport mysql.api as mysql

cdef extern from "mysqlmod.h":

    ctypedef struct _mysql_ConnectionObject:
        pass
    ctypedef struct _mysql_ResultObject:
        pass
    ctypedef struct _mysql_FieldObject:
        pass

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


                                                   




