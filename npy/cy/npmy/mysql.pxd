

cdef extern from "mysql.h":
    ## enums
    ## types
    ctypedef struct MYSQL:
        pass
    ctypedef struct MYSQL_RES:
        pass
    ctypedef struct MYSQL_FIELD:
        pass
    ctypedef char **MYSQL_ROW
    ## functions 
    cdef MYSQL_ROW       mysql_fetch_row(MYSQL_RES *result)
    cdef unsigned int    mysql_num_fields(MYSQL_RES *result)
