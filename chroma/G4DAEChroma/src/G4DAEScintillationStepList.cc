#include "G4DAEChroma/G4DAEScintillationStepList.hh"
#include "G4DAEChroma/G4DAEArray.hh"

const char* G4DAEScintillationStepList::TMPL = "DAESCINTILLATION_PATH_TEMPLATE" ;
const char* G4DAEScintillationStepList::SHAPE = "6,4" ;
const char* G4DAEScintillationStepList::KEY   = "XXX" ;

G4DAEScintillationStepList::G4DAEScintillationStepList( G4DAEArray* array ) : G4DAEArrayHolder(array) {}
G4DAEScintillationStepList::G4DAEScintillationStepList( std::size_t itemcapacity, float* data) : G4DAEArrayHolder( itemcapacity, data, SHAPE ) {}
G4DAEScintillationStepList::~G4DAEScintillationStepList() {}

void G4DAEScintillationStepList::Save(const char* evt, const char* key, const char* tmpl )
{
    m_array->Save<G4DAEScintillationStepList>(evt, key, tmpl);
}
void G4DAEScintillationStepList::SavePath(const char* path, const char* key)
{
    m_array->SavePath<G4DAEScintillationStepList>(path, key);
}


// statics : seem almost extraneous, but no so due to the tmpl default arguments 

std::string G4DAEScintillationStepList::GetPath( const char* evt, const char* tmpl)
{
    return G4DAEArray::GetPath(evt, tmpl);
}
G4DAEScintillationStepList* G4DAEScintillationStepList::Load(const char* evt, const char* key, const char* tmpl)
{
    return G4DAEArray::Load<G4DAEScintillationStepList>(evt, key, tmpl);
}
G4DAEScintillationStepList* G4DAEScintillationStepList::LoadPath(const char* path, const char* key)
{
    return G4DAEArray::LoadPath<G4DAEScintillationStepList>(path, key);
}



