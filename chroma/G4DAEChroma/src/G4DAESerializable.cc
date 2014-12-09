#include "G4DAEChroma/G4DAESerializable.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"

// do nothing default implementation, requiring no storage

void G4DAESerializable::SetLink(G4DAEMetadata* /*link*/)
{
}
G4DAEMetadata* G4DAESerializable::GetLink()
{
    return NULL ; 
}

// note no allocation needed below, rely on subclasses to provide
// that together with get/set overrides of above

G4DAEMetadata* G4DAESerializable::GetLastLink()
{
    G4DAEMetadata* last = NULL ;
    G4DAEMetadata* m = GetLink();
    while(m)
    {   
        last = m ;
        m = m->GetLink();
    } 
    return last ;
}

void G4DAESerializable::AddLink(G4DAEMetadata* link )
{
    G4DAEMetadata* last = GetLastLink();
    if(last)
    {
         last->SetLink(link);
    }
    else 
    {
         SetLink(link); 
    }
}


const char* G4DAESerializable::GetMagic()
{
    return NULL ; 
}






/*
// do nothing starting point for implementing G4DAESerializable

void G4DAESerializable::SaveToBuffer()
{
}
const char* G4DAESerializable::GetBufferBytes()
{
    return NULL ; 
}
std::size_t G4DAESerializable::GetBufferSize()
{
    return 0 ; 
}
void G4DAESerializable::DumpBuffer()
{
}
G4DAESerializable* G4DAESerializable::CreateOther(char* bytes, std::size_t size)
{
    return NULL ;
}

*/


