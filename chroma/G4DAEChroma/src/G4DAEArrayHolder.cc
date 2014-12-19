
#include "G4DAEChroma/G4DAEArrayHolder.hh"
#include "G4DAEChroma/G4DAEArray.hh"


G4DAEArrayHolder::G4DAEArrayHolder( G4DAEArray* array ) : m_array(array) {}


G4DAEArrayHolder::G4DAEArrayHolder( std::size_t itemcapacity, float* data, const char* shape )
{
   m_array = new G4DAEArray(itemcapacity, shape, data );
}

G4DAEArrayHolder::~G4DAEArrayHolder()
{
   delete m_array ;
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


