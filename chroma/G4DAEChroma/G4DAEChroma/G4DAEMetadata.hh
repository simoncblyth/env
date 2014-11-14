#ifndef G4DAEMETADATA_H
#define G4DAEMETADATA_H

#include <map>
#include <string>

#include "G4DAEChroma/G4DAESerializable.hh"

class G4DAEBuffer ;

class G4DAEMetadata : public G4DAESerializable {
public:
    static const std::string EMPTY ; 
public:
    G4DAEMetadata();
    virtual ~G4DAEMetadata();

public:
    void Set(const char* key, const char* val );
    std::string& Get(const char* key);

public:
    // debugging 
    void SetString(const char* str);

public:
   // G4DAESerializable
   virtual void SaveToBuffer();
   virtual const char* GetBufferBytes();
   virtual std::size_t GetBufferSize();
   virtual void DumpBuffer();
   virtual G4DAESerializable* CreateOther(char* bytes, std::size_t size);

   void SetLink(G4DAESerializable* link );
   G4DAESerializable* GetLink();


private:
    std::map<std::string,std::string> m_kv ;
    G4DAEBuffer* m_buffer ; 
    G4DAESerializable* m_link ;

};

#endif
