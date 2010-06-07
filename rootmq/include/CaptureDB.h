
#ifndef CAPTUREDB_H
#define CAPTUREDB_H

#include "sqlite3.h"



class CaptureDB { 
    /*
         Trivial C++ wrapping of SQLite3 ...
       
               http://souptonuts.sourceforge.net/readme_sqlite_tutorial.html
               http://souptonuts.sourceforge.net/code/simplesqlite3.c.html

         Command line usage of sqlite3 :

               sqlite3  $(env-home)/scons-out/dbg/tests/try.db  "create table AbtEvent (id INTEGER PRIMARY KEY,data TEXT);"   
               sqlite3 $(env-home)/scons-out/dbg/tests/try.db "select data from AbtEvent where id=0;"
               sqlite3 -noheader $(env-home)/scons-out/dbg/tests/try.db "select data from AbtEvent where id=1024;"

           TODO:
                table creation in here rather than relying on command line bootstrap
                 (need to work out how to do table existance introspection : probably a query agains sqlite_master_??) 
                 sqlite3  $(env-home)/scons-out/dbg/tests/try.db  "create table AbtEvent (id INTEGER PRIMARY KEY,data TEXT);"  
                          
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







