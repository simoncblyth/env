
#include "G4DAEChroma/G4DAEArrayHolder.hh"
#include "G4DAEChroma/G4DAEArray.hh"


G4DAEArrayHolder::G4DAEArrayHolder( G4DAEArray* array ) : m_array(array), m_link(NULL) {}


G4DAEArrayHolder::G4DAEArrayHolder( std::size_t itemcapacity, float* data, const char* shape ) : m_array(NULL), m_link(NULL)
{
   m_array = new G4DAEArray(itemcapacity, shape, data );
}

G4DAEArrayHolder::~G4DAEArrayHolder()
{
   delete m_array ;
   // delete m_link ; **NOT DELETING LINK : REGARDED AS WEAK REFERENCE**
}

void G4DAEArrayHolder::Print(const char* msg) const 
{
    if(m_array) m_array->Print(msg);
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




