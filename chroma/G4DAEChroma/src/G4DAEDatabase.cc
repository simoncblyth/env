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
        //cout << "G4DAEDatabase::G4DAEDatabase : INFO : envvar " << envvar << " opening path " << path << endl ; 
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

int G4DAEDatabase::Insert(G4DAEMetadata* meta, const char* name, const char* columns )
{
    if(!meta || !m_db) return -1;

    if(!name) name = meta->GetName();

    Map_t row  = meta->GetRowMap(columns);
    Map_t type = meta->GetTypeMap(columns);

    // std::map<std::string,std::string> iterator order is by key comparison, ie alphabetic
    // so need to pass columns to control order 

    m_db->Create(name, type, columns);  // create table if not existing 
    m_db->Insert(name, row);   

    return m_db->LastInsertRowId();
}

int G4DAEDatabase::Query(const char* sql )
{
    if(!m_db) return -2 ;
    return m_db->Exec(sql); 
}

int G4DAEDatabase::QueryI(const char* sql, int param )
{
    if(!m_db) return -2 ;
    return m_db->ExecI(sql, param); 
}



Map_t G4DAEDatabase::GetRow(std::size_t index)
{
    Map_t row ;
    if(m_db) row = m_db->GetRow(index);
    return row;
}

std::vector<long> G4DAEDatabase::GetIVec(const char* column, const char* sql)
{
    std::vector<long> ivec ;
    if(m_db) ivec = m_db->GetIVec(column, sql);
    return ivec ;
}



Map_t G4DAEDatabase::GetOne(const char* sql, int id)
{
    Map_t row ;
    int rc = this->QueryI(sql, id); 
    if( rc > -1 )
    {
       row = this->GetRow(0);
    }
    return row ; 
}


