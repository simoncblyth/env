#include "G4DAEChroma/G4DAESerializable.hh"

// do nothing default implementation, requiring no storage

void G4DAESerializable::SetLink(G4DAESerializable* link)
{
}
G4DAESerializable* G4DAESerializable::GetLink()
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


