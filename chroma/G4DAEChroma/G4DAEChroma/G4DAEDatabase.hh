#ifndef G4DAEDATABASE_H
#define G4DAEDATABASE_H

class Database ;
class G4DAEMetadata ; 

#include "G4DAEChroma/G4DAEMap.hh"

class G4DAEDatabase {
public:
    G4DAEDatabase(const char* envvar);
    virtual ~G4DAEDatabase();

public:
    // into DB
    int Insert(G4DAEMetadata* metadata, const char* name=NULL, const char* columns=NULL);

public:
    // low level DB access : with no DB table assumptions
    Map_t GetOne(const char* sql, int id);
    Map_t GetRow(std::size_t index=0);
    int Query(const char* sql );   // returns rowcount, or negative for error
    int QueryI(const char* sql, int param );  

private:
    Database* m_db ;

};


#endif

