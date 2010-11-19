"""
   Only wrap needed symbols 

"""
cdef extern from "mysql.h":
    enum enum_field_types:
        MYSQL_TYPE_DECIMAL
        MYSQL_TYPE_TINY
        MYSQL_TYPE_SHORT
        MYSQL_TYPE_LONG
        MYSQL_TYPE_FLOAT
        MYSQL_TYPE_DOUBLE
        MYSQL_TYPE_NULL   
        MYSQL_TYPE_TIMESTAMP
        MYSQL_TYPE_LONGLONG
        MYSQL_TYPE_INT24
        MYSQL_TYPE_DATE   
        MYSQL_TYPE_TIME
        MYSQL_TYPE_DATETIME
        MYSQL_TYPE_YEAR
        MYSQL_TYPE_NEWDATE
        MYSQL_TYPE_ENUM
        MYSQL_TYPE_SET
        MYSQL_TYPE_TINY_BLOB
        MYSQL_TYPE_MEDIUM_BLOB
        MYSQL_TYPE_LONG_BLOB
        MYSQL_TYPE_BLOB
        MYSQL_TYPE_VAR_STRING
        MYSQL_TYPE_STRING
        MYSQL_TYPE_GEOMETRY


    ctypedef unsigned long long my_ulonglong
    ctypedef char** MYSQL_ROW

    ctypedef struct MYSQL:
        pass
    ctypedef struct MYSQL_RES:
        pass
    ctypedef struct MYSQL_FIELD:
        pass

    MYSQL_ROW       mysql_fetch_row(MYSQL_RES *res)
    unsigned int    mysql_num_fields(MYSQL_RES *res)
    my_ulonglong    mysql_num_rows(MYSQL_RES *res)

 
