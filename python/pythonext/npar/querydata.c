#include "querydata.h"
#include <stdlib.h>
#include <stdio.h>
#include <sqlite3.h> 


char Type(int type)
{
   char rc = '?';
   switch(type)
   {
       case SQLITE_INTEGER:rc = 'i' ;break;
       case SQLITE_FLOAT:rc = 'f' ;break;
       case SQLITE_TEXT:rc = 't' ;break;
       case SQLITE_BLOB:rc = 'b' ;break;
       case SQLITE_NULL:rc = 'n' ;break;
   }
   return rc ;
}



//void* querydata( const char* sql, int* nrow, int* ncol, char* type )
int querydata( const char* envvar, const char* sql, int* nrow, int* ncol, char* type, float* fbuf, int fbufmax )
{
    //printf("SQLITE_VERSION   %s \n", SQLITE_VERSION );
    //printf("SQLITE_SOURCE_ID %s \n", SQLITE_SOURCE_ID );
    //printf("querydata sql %s \n", sql );

    sqlite3* m_db;
    sqlite3_stmt* m_statement ; 

    const char* path = getenv(envvar);
    if(path)
    {
        if( sqlite3_open(path,&m_db) != SQLITE_OK )
        {
            fprintf(stderr, "Can't open database at path %s: %s\n", path, sqlite3_errmsg(m_db));
            return -1 ; 
        }   
        //fprintf(stderr, "Opened DB at %s \n", path);
    }
    else
    {
        fprintf(stderr, "envvar pointing to sqlite3 db is not set %s \n", envvar);
        return -2 ;  
    }
  

    if( sqlite3_prepare_v2(m_db, sql, -1, &m_statement, 0) != SQLITE_OK )
    {   
        const char* err = sqlite3_errmsg(m_db);
        fprintf(stderr, "Database::Exec sqlite3_prepare_v2 error with sql %s : %s \n", sql, err );
        return -3 ; 
    }   



    int columns = sqlite3_column_count(m_statement) ;

    const int maxcol = 16 ;
    char types[maxcol] = {0};

    size_t count = 0 ; 
    size_t index = 0 ; 
    while(1)
    {
        int s = sqlite3_step(m_statement) ;
        if( s == SQLITE_ROW )
        {
            if(index >= fbufmax )
            {
               fprintf(stderr, "buffer not big enough, array truncated : %d \n", fbufmax );
               break ;
            }


            for(int c = 0; c < columns; c++)
            {
                //const char* name = sqlite3_column_name(statement, c);
                //const char* text = (const char*)sqlite3_column_text(statement, c);
                //printf("%s : %s \n", name, text );

                if(count == 0 && c < maxcol) types[c] = Type(sqlite3_column_type(m_statement, c));

                index = count*columns + c ;  
 
                switch(types[c])
                {
                   case 'f': 
                             *(fbuf + index) = sqlite3_column_double(m_statement, c );  
                             break;     
                   case 'i': 
                             *(fbuf + index) = sqlite3_column_double(m_statement, c );  
                             break; 
                   default:  
                             break;
                }
            }
            count++;

       } else if( s == SQLITE_DONE ) {

           break;

       } else {
            fprintf (stderr, "sqlite3_step Failed.\n");
            return -5 ;
       }

    }
    sqlite3_finalize(m_statement);


    *type = 'f' ; 
    *ncol = columns ; 
    *nrow = count ;


    if(m_db)
    {
        //fprintf(stderr, "Closing DB at path %s \n", path);
        sqlite3_close(m_db); 
    }

    return 0 ; 
}


