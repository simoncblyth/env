/*
Build and run this main with::

   rapsqlite-testjs

Rebuild the underlying libs with either::

   rapsqlite-build 
   rapsqlite-build-full

   cjs-build
   cjs-build-full
 
   
*/
#include "RapSqlite/Database.hh"
#include "cJSON/JS.hh"


void row_dumping( Database& db )
{
    Map_t& type = db.GetRowType();
    db.DumpMap("rowtype", type);
    printf("rowspec %s\n", db.GetRowSpec().c_str());

    VMap_t& rows = db.GetRows();
    VMap_t::iterator it = rows.begin();
    while( it != rows.end() )
    {
        db.DumpMap("rows", *it );
        ++it ;
    }
}


int main()
{
    Database db("DBPATH");

    int nrow = db.Exec("select * from mocknuwa;");
    if( nrow < 0 )  return -1 ;

    //row_dumping(db);

    Map_t row = db.GetRow(0);

    JS* js = new JS();
    js->AddMap("query", row ); 
    js->Print();

    std::string pretty = js->AsString(true);    
    printf("pretty %s\n", pretty.c_str());

    std::string ugly = js->AsString(false);    
    printf("ugly %s\n", ugly.c_str());


    return 0 ;
}


