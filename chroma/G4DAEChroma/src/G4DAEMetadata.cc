#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEBuffer.hh"
#include "G4DAEChroma/G4DAETime.hh"
#include <string.h>
#include <iostream>
#include <stdio.h>

#include "cJSON/js.hh"

using namespace std ; 

const std::string G4DAEMetadata::EMPTY = "empty" ; 

const char* G4DAEMetadata::TIMEFORMAT = "%Y-%m-%d %H:%M:%S" ; 

double G4DAEMetadata::RealTime()
{
   return getRealTime();
}

char* G4DAEMetadata::TimeStampLocal()
{
   return now(G4DAEMetadata::TIMEFORMAT, 20, 0); 
}

char* G4DAEMetadata::TimeStampUTC()
{
   return now(G4DAEMetadata::TIMEFORMAT, 20, 1); 
}



G4DAEMetadata::G4DAEMetadata(Map_t& map, const char* name) : m_buffer(NULL), m_link(NULL), m_js(NULL)
{
    SetName("meta");
    SetString("{}");
    std::string s = GetString();
    m_js = new JS(s.c_str());
    m_js->AddMap(name, map);
}


G4DAEMetadata::G4DAEMetadata(const char* str) : m_buffer(NULL), m_link(NULL), m_js(NULL)
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


// setting into the JSON tree for existing top level object "name" 
void G4DAEMetadata::SetKV(const char* name, const char* key, const char* val)
{
    if(!m_js) return ;
    m_js->SetKV(name, key, val);
}

void G4DAEMetadata::SetKV(const char* name, const char* key, int val)
{
    const int len = 10 ;
    char sval[len];
    snprintf(sval, len, "%d", val);
    SetKV(name, key, sval);
}

void G4DAEMetadata::Set(const char* key, int val )
{
    const int len = 10 ;
    char sval[len];
    snprintf(sval, len, "%d", val);
    Set(key, sval);
}


void G4DAEMetadata::Set(const char* key, const char* val )
{
    if(!key || !val){
        printf("G4DAEMetadata::Set null key or val \n");
        return ;
    }

    //printf("G4DAEMetadata::Set key %s val %s \n", key,val);
 
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

void G4DAEMetadata::AddMap(const char* name, Map_t& map)
{
    if(!m_js) return ;
    m_js->AddMap(name, map);
} 

Map_t G4DAEMetadata::GetMap(const char* wanted)
{
    Map_t emp ;
    return m_js ? m_js->GetMap(wanted) : emp ;
} 

void G4DAEMetadata::DumpMap(Map_t& map, const char* msg)
{
    JS::DumpMap(map, msg);
} 


void G4DAEMetadata::PrintMap(const char* msg)
{
    if(!m_js) return ;
    m_js->PrintMap(msg);
} 


void G4DAEMetadata::SaveToBuffer()
{
    if(!m_js) return ;
    std::string str = m_js->AsString();
    //printf("G4DAEMetadata::SaveToBuffer str %s \n", str.c_str() );
    SetString( str.c_str());
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

Map_t G4DAEMetadata::GetRowMap(const char* columns) 
{
   // single row values for inserting into DB table
    Map_t rmap ;
    if(m_js) rmap = m_js->CreateRowMap(columns);
    return rmap ;
}

Map_t G4DAEMetadata::GetTypeMap(const char* columns) 
{
   // this defines columns and types of a DB table
    Map_t tmap ;
    if(m_js) tmap = m_js->CreateTypeMap(columns);
    return tmap ;
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



