/*
Build and run with *rapsqlite-test*
*/

#include "RapSqlite/Database.hh"

int main()
{
    Database db("DBPATH");
    db.Introspect();
    db.Create("D", "x:int,y:string,z:float,w:float");
    db.Insert("D","x:1,y:hello,z:1.1,w:2.2");
    db.Select("D");
    db.DumpTables();
}


