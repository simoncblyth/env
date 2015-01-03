
#include "G4DAEChroma/G4DAEArrayHolder.hh"
#include "G4DAEChroma/G4DAEArray.hh"

#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEMap.hh"



// CAUTION: copy ctor that just steals pointers without copying 
G4DAEArrayHolder::G4DAEArrayHolder( G4DAEArrayHolder* other ) : m_array(other->GetArray()), m_link(other->GetLink()) {}

G4DAEArrayHolder::G4DAEArrayHolder( G4DAEArrayHolder* other, int start, int stop, int step) : m_array(NULL), m_link(other->GetLink()) 
{
    m_array = new G4DAEArray( other->GetArray(), start, stop, step );
}


G4DAEArrayHolder::G4DAEArrayHolder( G4DAEArray* array ) : m_array(array), m_link(NULL) {}

G4DAEArrayHolder::G4DAEArrayHolder( G4DAEArray* array, int start, int stop, int step ) : m_array(NULL), m_link(NULL) 
{
    m_array = new G4DAEArray( array, start, stop, step );
}




G4DAEArrayHolder::G4DAEArrayHolder( std::size_t itemcapacity, float* data, const char* shape ) : m_array(NULL), m_link(NULL)
{
   m_array = new G4DAEArray(itemcapacity, shape, data );
}

G4DAEArrayHolder::~G4DAEArrayHolder()
{
   delete m_array ;
   // delete m_link ; **NOT DELETING LINK : REGARDED AS WEAK REFERENCE**
}


void G4DAEArrayHolder::SetArray(G4DAEArray* array )
{
   delete m_array ; 
   m_array = array ; 
} 

G4DAEArray* G4DAEArrayHolder::GetArray()
{
   return m_array ; 
} 

void G4DAEArrayHolder::Print(const char* msg) const 
{
    if(m_array) m_array->Print(msg);
}




void G4DAEArrayHolder::CreateLink() 
{
    if(!m_link)
    {
        m_link = new G4DAEMetadata("{}") ;
    }
}
void G4DAEArrayHolder::AddMap(const char* name, Map_t& meta) 
{
    CreateLink();
    m_link->AddMap(name, meta); 
}
void G4DAEArrayHolder::SetKV(const char* name, const char* key, const char* val)
{
    CreateLink();
    m_link->SetKV(name, key, val);
}
void G4DAEArrayHolder::SetKV(const char* name, const char* key, int ival)
{
    CreateLink();
    m_link->SetKV(name, key, ival);
}





std::size_t G4DAEArrayHolder::GetCount() const {
    return m_array ? m_array->GetSize() : 0 ;
}

std::string G4DAEArrayHolder::GetDigest() const {
    return m_array ? m_array->GetDigest() : "" ;
}

void G4DAEArrayHolder::ClearAll() {
    if(m_array) m_array->ClearAll();
}

float* G4DAEArrayHolder::GetNextPointer() {
    return m_array ? m_array->GetNextPointer() : NULL  ;
}

float* G4DAEArrayHolder::GetItemPointer(std::size_t index) {
    return m_array ? m_array->GetItemPointer(index) : NULL  ;
}




// G4DAESerializable
G4DAEArrayHolder*  G4DAEArrayHolder::CreateOther(char* buffer, std::size_t buflen)
{
    if(!m_array) return NULL ;
    G4DAEArray* array = m_array->CreateOther(buffer, buflen);
    return new G4DAEArrayHolder(array);
}

void G4DAEArrayHolder::SaveToBuffer()
{
   if(!m_array) return ;
   m_array->SaveToBuffer();
}
void G4DAEArrayHolder::DumpBuffer()
{
   if(!m_array) return ;
   m_array->DumpBuffer();
}
const char* G4DAEArrayHolder::GetBufferBytes()
{
    return m_array ? m_array->GetBufferBytes() : NULL ;
}
std::size_t G4DAEArrayHolder::GetBufferSize()
{
    return m_array ? m_array->GetBufferSize() : 0 ; 
}
const char* G4DAEArrayHolder::GetMagic()
{
    return m_array ? m_array->GetMagic() : NULL ;
}
void G4DAEArrayHolder::SetLink(G4DAEMetadata* link )
{
    m_link = link ;
}
G4DAEMetadata* G4DAEArrayHolder::GetLink()
{
    return m_link ;
}




