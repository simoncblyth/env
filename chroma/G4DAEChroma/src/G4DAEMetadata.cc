#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEBuffer.hh"
#include <string.h>
#include <iostream>

using namespace std ; 

const std::string G4DAEMetadata::EMPTY = "empty" ; 

G4DAEMetadata::G4DAEMetadata(const char* str ) : m_buffer(NULL), m_link(NULL)
{
   SetString(str);
}
G4DAEMetadata::G4DAEMetadata(G4DAEBuffer* buffer) : m_buffer(buffer), m_link(NULL) 
{
    m_kv[EMPTY] = "" ; 
}
G4DAEMetadata::~G4DAEMetadata()
{
    delete m_buffer ; 
    //delete m_link ; **NOT DELETING LINK : REGARD AS WEAK REFERENCE**
}

void G4DAEMetadata::Set(const char* key, const char* val )
{
    string k(key);
    string v(val);
    m_kv[k] = v ;
}
std::string& G4DAEMetadata::Get(const char* key)
{
    string k(key);
    return ( m_kv.find(k) == m_kv.end()) ? m_kv[EMPTY] : m_kv[k] ;
}


void G4DAEMetadata::SetString(const char* str)
{
    delete m_buffer ; 
    m_buffer = new G4DAEBuffer(strlen(str), const_cast<char*>(str)); 
}     

std::string G4DAEMetadata::GetString() const 
{
    return m_buffer ? std::string(m_buffer->GetBytes(), m_buffer->GetSize()) : std::string() ; 
}


void G4DAEMetadata::Print(const char* msg) const
{
    cout << msg 
         << " str: " << GetString() << endl ; 

    if(m_link) m_link->Print(msg);
}




void G4DAEMetadata::SaveToBuffer()
{
    // not needed when use SetString
}

const char* G4DAEMetadata::GetBufferBytes()
{
   return m_buffer ? m_buffer->GetBytes() : NULL ;
}
std::size_t G4DAEMetadata::GetBufferSize()
{
   return m_buffer ? m_buffer->GetSize() : 0 ;
}
void G4DAEMetadata::DumpBuffer()
{
   printf("G4DAEMetadata::DumpBuffer \n");
   if(m_buffer) m_buffer->Dump() ;
}



G4DAEMetadata* G4DAEMetadata::CreateOther(char* buffer, std::size_t buflen)
{
    G4DAEBuffer* buf = new G4DAEBuffer(buflen, buffer);
    return new G4DAEMetadata(buf);
    // where/when to deserialize into the map : maybe json parser for this
}


void G4DAEMetadata::SetLink(G4DAEMetadata* link )
{
    m_link = link ;
}
G4DAEMetadata* G4DAEMetadata::GetLink()
{
    return m_link ;
}





   



