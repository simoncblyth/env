/*
  cc $(testsqlite-sdir)/testsqlite.c -L/opt/local/lib -lsqlite3 -o $LOCAL_BASE/env/bin/testsqlite  && testsqlite /tmp/testsqlite.db

  http://www.sqlite.org/cintro.html
  http://www.sqlite.org/c3ref/exec.html

  http://www.tutorialspoint.com/sqlite/sqlite_c_cpp.htm


*/

#include <stdlib.h>
#include <stdio.h>
#include <sqlite3.h> 
#include <assert.h> 


static int callback(void *NotUsed, int argc, char **argv, char **azColName){
   int i;
   for(i=0; i<argc; i++)
   {
      printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
   }
   printf("\n");
   return 0;
}



int main(int argc, char** argv)
{

   assert(argc > 1);
   const char* path = argv[1] ;  

   sqlite3 *db;
   char *zErrMsg = 0;
   int rc;
   char *sql;

   rc = sqlite3_open(path, &db);

   if( rc )
   {
       fprintf(stderr, "Can't open database at path %s: %s\n", path, sqlite3_errmsg(db));
       exit(0);
   }
   else
   {
       fprintf(stderr, "Opened database at %s successfully\n", path );
   }


   /* Create SQL statement */
   sql = "CREATE TABLE COMPANY("  \
         "ID INT PRIMARY KEY     NOT NULL," \
         "NAME           TEXT    NOT NULL," \
         "AGE            INT     NOT NULL," \
         "ADDRESS        CHAR(50)," \
         "SALARY         REAL );";


   rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);

   if( rc != SQLITE_OK )
   {
       fprintf(stderr, "SQL error: %s\n", zErrMsg);
       sqlite3_free(zErrMsg);
   }
   else
   {
      fprintf(stdout, "Table created successfully\n");
   }


   sqlite3_close(db);
   return 0 ;
}




