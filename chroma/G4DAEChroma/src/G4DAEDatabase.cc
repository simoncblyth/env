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

void G4DAEDatabase::Insert(G4DAEMetadata* metadata)
{
    if(!metadata || !m_db) return;
    string meta = metadata->GetString();
    cout << "G4DAEDatabase::Insert " << meta << endl ; 
}



