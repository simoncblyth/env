#ifndef G4DAEDATABASE_H
#define G4DAEDATABASE_H

#include <string>
#include <map>
typedef std::map<std::string,std::string>  Map_t ;

class Database ;
class G4DAEMetadata ; 

class G4DAEDatabase {
public:
    G4DAEDatabase(const char* envvar);
    virtual ~G4DAEDatabase();

public:
    // into DB
    void Insert(G4DAEMetadata* metadata);
public:
    // from DB
    Map_t GetRow(std::size_t index=0);
    int Query(const char* sql );   // returns rowcount, or negative for error
    int QueryI(const char* sql, int param );  

private:
    Database* m_db ;

};


#endif

