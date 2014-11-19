#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEBuffer.hh"
#include <string.h>
#include <iostream>

#include "cJSON/js.hh"

using namespace std ; 

const std::string G4DAEMetadata::EMPTY = "empty" ; 

G4DAEMetadata::G4DAEMetadata(const char* str ) : m_buffer(NULL), m_link(NULL), m_js(NULL)
{
    SetString(str);
    SetName("meta");
    m_js = new JS(str);
}
G4DAEMetadata::G4DAEMetadata(G4DAEBuffer* buffer) : m_buffer(buffer), m_link(NULL), m_js(NULL) 
{
    m_kv[EMPTY] = "" ; 
    SetName("meta");
    std::string s = GetString();
    if(!s.empty()) m_js = new JS(s.c_str());
}
G4DAEMetadata::~G4DAEMetadata()
{
    delete m_buffer ; 
    //delete m_link ; **NOT DELETING LINK : REGARD AS WEAK REFERENCE**
    delete m_js ; 
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

void G4DAEMetadata::Merge(const char* name)
{
    if(!m_js) return ;
    m_js->AddMap(name, m_kv);
}

void G4DAEMetadata::Print(const char* msg) const
{
    //if(m_link) m_link->Print(msg);
    if(m_js){
        m_js->Print(msg); 
    }
    else{
        cout << msg << GetString() << endl ; 
    }
}
void G4DAEMetadata::PrintToFile(const char* path) const
{
    if(!m_js) return ;
    m_js->PrintToFile(path);
}

void G4DAEMetadata::SetName(const char* name)
{
    m_name.assign(name);
}

const char* G4DAEMetadata::GetName()
{
    return m_name.c_str();
}

Map_t G4DAEMetadata::GetRowMap() 
{
    Map_t rmap ;
    if(m_js) rmap = m_js->CreateRowMap();
    return rmap ;
}

Map_t G4DAEMetadata::GetTypeMap() 
{
    Map_t tmap ;
    if(m_js) tmap = m_js->CreateTypeMap();
    return tmap ;
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



