


#include <cstdlib>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

//#include <stdio.h>
//#include <string.h>
#include "CaptureDB.h"

CaptureDB::CaptureDB( const char* dbpath ){
   m_datum = NULL ;
   m_dbg = 0 ;
   int rc = sqlite3_open(dbpath, &m_db); 
   if( rc ){
     fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(m_db));
     sqlite3_close(m_db);
     exit(1);
   }
}

int CaptureDB::callback( void* me , int argc, char **argv, char **azColName){
   CaptureDB* self = (CaptureDB*)me ;
   int i;
   for(i=0; i<argc; i++){
      if(self->m_dbg > 1) printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
      if(i==0){
          char* s = argv[i];
          size_t n = strlen(s);
          self->m_datum  = new char[n+1];
          strcpy( self->m_datum , s );
      }
   }   
   if(self->m_dbg > 1) printf("\n");
   return 0;
}


void CaptureDB::Exec( const char* sql ){
   int rc = sqlite3_exec(m_db, sql , callback, this, &zErrMsg);
   if( rc!=SQLITE_OK ){
       fprintf(stderr, "SQL error: %s\n", zErrMsg);
   }
}

const char* CaptureDB::Get( const char* table, int i)
{
    stringstream ss ;
    ss << "select data from " << table << " where id=" << i << " ;" ;
    Exec( ss.str().c_str() );
    return m_datum ;
}

const char* CaptureDB::GetLast()
{
    return m_datum ;
}

CaptureDB::~CaptureDB(){
    sqlite3_close(m_db);
}



