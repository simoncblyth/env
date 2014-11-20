#ifndef G4DAEMETADATA_H
#define G4DAEMETADATA_H

#include <map>
#include <string>

#include "G4DAEChroma/G4DAESerializable.hh"

class G4DAEBuffer ;
class G4DAEMetadata ; 
class JS ; 


class G4DAEMetadata : public G4DAESerializable {
public:
    typedef std::map<std::string,std::string> Map_t ;
    static const std::string EMPTY ; 
public:
    G4DAEMetadata(const char* str);
    G4DAEMetadata(G4DAEBuffer* buffer=NULL);
    virtual ~G4DAEMetadata();

public:
    // set,get manual key value pairs
    void Set(const char* key, const char* val );
    std::string& Get(const char* key);

    // merge JSON tree parsed from initial string/buffer together 
    // with manual key/values map under new JSON top level object 
    // named by the argument
    void Merge(const char* name); 

    // incorporate key, values from map argument into contained JSON tree in 
    // new top level JSON object named by the argument 
    void AddMap(const char* name, Map_t& map); 

public:
    // debugging 
    void PrintToFile(const char* path) const;
    void Print(const char* msg="G4DAEMetadata::Print") const;
    void SetString(const char* str);
    std::string GetString() const;

public:
   // G4DAESerializable
   virtual void SaveToBuffer();
   virtual const char* GetBufferBytes();
   virtual std::size_t GetBufferSize();
   virtual void DumpBuffer();
   virtual G4DAEMetadata* CreateOther(char* bytes, std::size_t size);

   void SetLink(G4DAEMetadata* link );
   G4DAEMetadata* GetLink();

public:
    // Database insertion
    void SetName(const char* name);
    const char* GetName();
    Map_t GetRowMap();
    Map_t GetTypeMap();

private:
    Map_t m_kv ;
    G4DAEBuffer* m_buffer ; 
    G4DAEMetadata* m_link ;
    JS* m_js ; 
    std::string m_name ; 

};

#endif
