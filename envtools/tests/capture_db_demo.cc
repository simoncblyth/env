
#include <sys/stat.h>

#include "Capture.h"
#include "CaptureDB.h"


int create_db( const char* path )
{
    string table = "Demo" ;
    cout << "create_db : creating " << path << " and table :" << table << endl ;
    CaptureDB db(path);

    stringstream ss ;
    ss << "create table " << table << " (id INTEGER PRIMARY KEY,data TEXT);" ;
    db.Exec(ss.str().c_str());  

    string key = "42" ;
    string found ;
    {
        Capture c ;
        cout << "CATCH THIS " << endl ;
        found = c.Gotcha();
    }
    ss.str(""); 
    ss << "insert into " << table << " ('id','data') values (" << key << ", '" << found << "' );" ;    
    db.Exec(ss.str().c_str());  
    return EXIT_SUCCESS ;
}

int read_db( const char* path )
{
    CaptureDB db(path);
    cout << "read_db : "  << db.Get("Demo", 42) << endl ;
    return EXIT_SUCCESS ;
} 


int main()
{
    const char* path = "capture_db_demo.db" ;
    struct stat s_buf;
    int rc = stat(path,&s_buf) ;
    if( -1 == rc ){
        cout << "capture_db_demo : DB does not exist : " << path << endl ;
        return create_db(path);
    } else if ( 0 == rc ){
        cout << "capture_db_demo : DB exists already : " << path << " ... delete it and re-run to recreate " << endl ;
        return read_db(path);
    }
    return EXIT_SUCCESS ;
}
