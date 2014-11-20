/*
Build and run this main with::

   rapsqlite-test

Rebuild the underlying lib with either::

   rapsqlite-build 
   rapsqlite-build-full
   
*/
#include "RapSqlite/Database.hh"

void create_insert(Database& db)
{
    db.Create("D", "x:int,y:string,z:float,w:float");
    db.Insert("D","x:1,y:hello,z:1.1,w:2.2");
    db.Select("D");
}

void test_exec(Database& db )
{
    int nrow = db.Exec("select * from mocknuwa;");
    if( nrow < 0 )  return ;

    Map_t& type = db.GetRowType();
    db.DumpMap("rowtype", type);
    printf("rowspec %s\n", db.GetRowSpec().c_str());

    VMap_t& rows = db.GetRows();
    for(VMap_t::iterator it=rows.begin() ; it != rows.end() ; it++ ) db.DumpMap("rows", *it );
}


int main()
{
    Database db("DBPATH");
    //db.SetDebug(3);
    db.DumpTables();

    test_exec(db);
}


