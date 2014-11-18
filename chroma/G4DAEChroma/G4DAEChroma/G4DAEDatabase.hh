#ifndef G4DAEDATABASE_H
#define G4DAEDATABASE_H

class Database ;
class G4DAEMetadata ; 

class G4DAEDatabase {
public:
    G4DAEDatabase(const char* envvar);
    virtual ~G4DAEDatabase();

public:
    void Insert(G4DAEMetadata* metadata);

private:
    Database* m_db ;

};


#endif

