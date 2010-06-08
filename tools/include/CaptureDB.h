
#ifndef CAPTUREDB_H
#define CAPTUREDB_H

#include "sqlite3.h"

class CaptureDB { 
    /*
         Trivial C++ wrapping of SQLite3 ...
       
               http://souptonuts.sourceforge.net/readme_sqlite_tutorial.html
               http://souptonuts.sourceforge.net/code/simplesqlite3.c.html
                          
           TODO : consider adoption of one of the many  C++ wrappers      
              sqlite3x-
              sq3- 
    
    */
   private:
      sqlite3* m_db ;
      char* zErrMsg ;
      char* m_datum  ;
      int m_dbg ;
      static int callback( void* me , int argc, char **argv, char **azColName);
      
   public:
       CaptureDB( const char* dbpath );
       const char* Get( const char* table, int i);
       const char* GetLast();
       void Exec( const char* sql );
       ~CaptureDB();
    
};

#endif







