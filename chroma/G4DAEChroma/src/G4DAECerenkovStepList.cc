#include "G4DAEChroma/G4DAECerenkovStepList.hh"
#include "G4DAEChroma/G4DAEArray.hh"

const char* G4DAECerenkovStepList::TMPL = "DAECERENKOV_PATH_TEMPLATE" ;
const char* G4DAECerenkovStepList::SHAPE = "6,4" ;
const char* G4DAECerenkovStepList::KEY   = "CSL" ;

G4DAECerenkovStepList::G4DAECerenkovStepList( G4DAEArray* array ) : G4DAEArrayHolder(array) {}
G4DAECerenkovStepList::G4DAECerenkovStepList( std::size_t itemcapacity, float* data) : G4DAEArrayHolder( itemcapacity, data, SHAPE ) {}
G4DAECerenkovStepList::~G4DAECerenkovStepList() {}

void G4DAECerenkovStepList::Save(const char* evt, const char* key, const char* tmpl )
{
    m_array->Save<G4DAECerenkovStepList>(evt, key, tmpl);
}
void G4DAECerenkovStepList::SavePath(const char* path, const char* key)
{
    m_array->SavePath<G4DAECerenkovStepList>(path, key);
}


// statics : seem almost extraneous, but no so due to the tmpl default arguments 

std::string G4DAECerenkovStepList::GetPath( const char* evt, const char* tmpl)
{
    return G4DAEArray::GetPath(evt, tmpl);
}
G4DAECerenkovStepList* G4DAECerenkovStepList::Load(const char* evt, const char* key, const char* tmpl)
{
    return G4DAEArray::Load<G4DAECerenkovStepList>(evt, key, tmpl);
}
G4DAECerenkovStepList* G4DAECerenkovStepList::LoadPath(const char* path, const char* key)
{
    return G4DAEArray::LoadPath<G4DAECerenkovStepList>(path, key);
}



