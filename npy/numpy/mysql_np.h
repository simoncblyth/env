#include "mysql.h"

static const char* NP_I1 = "i1" ; 
static const char* NP_I2 = "i2" ; 
static const char* NP_I4 = "i4" ;
static const char* NP_I8 = "i4" ;
static const char* NP_F4 = "f4" ; 
static const char* NP_F8 = "f8" ; 
static const char* NP_DT = "datetime64[us]" ; 
static const char* NP_ST = "|S%d" ; 

const char* mysql2np( int mt ){
   const char* npt = NULL ; 
   switch( mt ){

       case MYSQL_TYPE_TINY: 
          npt = NP_I1 ; 
          break ; 
       case MYSQL_TYPE_SHORT: 
          npt = NP_I2 ; 
          break ; 
       case MYSQL_TYPE_ENUM: 
       case MYSQL_TYPE_SET: 
       case MYSQL_TYPE_LONG: 
          npt = NP_I4 ; 
          break ; 
       case MYSQL_TYPE_LONGLONG: 
       case MYSQL_TYPE_INT24: 
          npt = NP_I8 ; 
          break ; 

       case MYSQL_TYPE_DECIMAL: 
       case MYSQL_TYPE_DOUBLE: 
          npt = NP_F8 ; 
          break ; 
       case MYSQL_TYPE_FLOAT: 
          npt = NP_F4 ; 
          break ; 
          
       case MYSQL_TYPE_DATE: 
       case MYSQL_TYPE_TIME: 
       case MYSQL_TYPE_DATETIME: 
       case MYSQL_TYPE_YEAR: 
       case MYSQL_TYPE_NEWDATE: 
       case MYSQL_TYPE_TIMESTAMP: 
          npt = NP_DT ; 
          break ; 
 
       case MYSQL_TYPE_VAR_STRING:
       case MYSQL_TYPE_STRING:
          npt = NP_ST ; 
          break ; 

   }
   return npt ;
}








