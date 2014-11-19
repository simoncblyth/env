#include "G4DAEChroma/G4DAEDatabase.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "RapSqlite/Database.hh"


#include <string>
#include <iostream>

typedef G4DAEMetadata::Map_t  Map_t ;
using namespace std ; 

G4DAEDatabase::G4DAEDatabase(const char* envvar) : m_db(NULL) 
{
    const char* path = Database::Path(envvar);
    if( path )
    {
        cout << "G4DAEDatabase::G4DAEDatabase : INFO : envvar " << envvar << " opening path " << path << endl ; 
        m_db = new Database(envvar);
    }
    else
    {
        cout << "G4DAEDatabase::G4DAEDatabase : WARNING : envvar " << envvar << " not defined : no db operations "  << endl ; 
    } 
}
G4DAEDatabase::~G4DAEDatabase()
{
    delete m_db ; 
}

void G4DAEDatabase::Insert(G4DAEMetadata* meta)
{
    if(!meta || !m_db) return;

    //string meta = metadata->GetString();
    Map_t row  = meta->GetRowMap();
    Map_t type = meta->GetTypeMap();

    m_db->Create(meta->GetName(), type);
    m_db->Insert(meta->GetName(), row);

}



