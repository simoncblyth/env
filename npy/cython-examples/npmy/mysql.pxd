

cdef extern from "mysql.h":
    ctypedef struct MYSQL:
        pass
    ctypedef struct MYSQL_RES:
        pass
    ctypedef struct MYSQL_FIELD:
        pass
    ctypedef struct MYSQL_ROW:
        pass

