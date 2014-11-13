#include "G4DAEChroma/G4DAEArrayHolder.hh"
#include "G4DAEChroma/G4DAEArray.hh"

G4DAEArrayHolder::G4DAEArrayHolder( G4DAEArray* arr ) : m_array(arr) {}

G4DAEArrayHolder::G4DAEArrayHolder( std::size_t itemcapacity, float* data, const char* shape ) 
{
    m_array = new G4DAEArray( itemcapacity, shape, data );
}
G4DAEArrayHolder::~G4DAEArrayHolder()
{
    delete m_array ; 
}


