#include "G4DAEChroma/G4DAEHitList.hh"
#include "G4DAEChroma/G4DAEArray.hh"

const char* G4DAEHitList::TMPL = "DAEHIT_PATH_TEMPLATE" ;
const char* G4DAEHitList::SHAPE = "8,3" ;
const char* G4DAEHitList::KEY   = "XXX" ;

G4DAEHitList::G4DAEHitList( G4DAEArray* array ) : G4DAEArrayHolder(array) {}
G4DAEHitList::G4DAEHitList( std::size_t itemcapacity, float* data) : G4DAEArrayHolder( itemcapacity, data, SHAPE ) {}
G4DAEHitList::~G4DAEHitList() {}


void G4DAEHitList::AddHit( G4DAEHit& hit )
{
    hit.Serialize(GetNextPointer());
}




// looks like could remove duplication below by moving to 
// G4DAEArrayHolder base but not so easy as many statics and the tmpl arguments
// ... probably the way ahead is for G4DAEArray to have SHAPE/TMPL as 
// dynamic properties : but then persistency gets more complicated  
// ... nope that doesnt fly need to be able to statically load so need the 
// tmpl to be available without instance

void G4DAEHitList::Save(const char* evt, const char* key, const char* tmpl )
{
    //G4DAEArrayHolder::Save(evt, key, tmpl);
    m_array->Save<G4DAEHitList>(evt, key, tmpl);
}
void G4DAEHitList::SavePath(const char* path, const char* key)
{
    //G4DAEArrayHolder::SavePath(path, key );
    m_array->SavePath<G4DAEHitList>(path, key);
}


// statics : seem almost extraneous, but no so due to the tmpl default arguments 

std::string G4DAEHitList::GetPath( const char* evt, const char* tmpl)
{
    return G4DAEArray::GetPath(evt, tmpl);
}
G4DAEHitList* Load(const char* evt, const char* key, const char* tmpl)
{
    return G4DAEArray::Load<G4DAEHitList>(evt, key, tmpl);
}
G4DAEHitList* LoadPath(const char* path, const char* key)
{
    return G4DAEArray::LoadPath<G4DAEHitList>(path, key);
}



