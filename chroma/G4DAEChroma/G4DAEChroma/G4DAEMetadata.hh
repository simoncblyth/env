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
    static const char* MAGIC ; 
    static const std::string EMPTY ; 
    static const char* TIMEFORMAT ; 
    static char* TimeStampLocal();
    static char* TimeStampUTC();
    static double RealTime();
    static G4DAEMetadata* CreateFromBuffer(char* bytes, std::size_t size);

public:
    G4DAEMetadata(Map_t& map, const char* name);
    G4DAEMetadata(const char* str );
    G4DAEMetadata(G4DAEBuffer* buffer=NULL);
    virtual ~G4DAEMetadata();

public:
    // set,get manual key value pairs
    void Set(const char* key, int val );
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
    // setting into the JSON tree for existing top level object "name" 
    void SetKV(const char* name, const char* key, const char* val);
    void SetKV(const char* name, const char* key, int val );

public:
    // read from JSON tree
    Map_t GetMap(const char* wanted);
    static void DumpMap(Map_t& map, const char* msg);

public:
    // debugging 
    void PrintToFile(const char* path) const;
    void Print(const char* msg="G4DAEMetadata::Print") const;
    void PrintMap(const char* msg="G4DAEMetadata::PrintMap");
    void SetString(const char* str);
    std::string GetString() const;

public:
   // G4DAESerializable
   virtual void SaveToBuffer();
   virtual const char* GetBufferBytes();
   virtual std::size_t GetBufferSize();
   virtual void DumpBuffer();
   virtual G4DAEMetadata* CreateOther(char* bytes, std::size_t size);
   const char* GetMagic();


   void SetLink(G4DAEMetadata* link );
   G4DAEMetadata* GetLink();

public:
    // Database insertion
    void SetName(const char* name);
    const char* GetName();
    //G4DAEMetadata* SubMeta(const char* name, const char* columns); 
    Map_t GetRowMap(const char* columns=NULL);
    Map_t GetTypeMap(const char* columns=NULL);

private:
    Map_t m_kv ;
    G4DAEBuffer* m_buffer ; 
    G4DAEMetadata* m_link ;
    JS* m_js ; 
    std::string m_name ; 

};

#endif
