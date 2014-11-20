#include "G4DAEChroma/G4DAEDatabase.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "RapSqlite/Database.hh"


#include <string>
#include <iostream>

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

    const char* name = meta->GetName();
    Map_t row  = meta->GetRowMap();
    Map_t type = meta->GetTypeMap();

    m_db->Create(name, type);  // create table if not existing 
    m_db->Insert(name, row);   

}

int G4DAEDatabase::Query(const char* sql )
{
    if(!m_db) return -2 ;
    return m_db->Exec(sql); 
}

Map_t G4DAEDatabase::GetRow(std::size_t index)
{
    Map_t row ;
    if(m_db) row = m_db->GetRow(index);
    return row;
}



