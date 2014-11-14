#include "G4DAEChroma/G4DAEBuffer.hh"
#include "G4DAEChroma/G4DAECommon.hh"

using namespace std ; 


G4DAEBuffer::G4DAEBuffer( size_t size, char* bytes ) : m_size(size), m_bytes(bytes)
{
   if(!size) return ;
   m_size = size ;
   m_bytes = new char[size] ;
   if(bytes) memcpy( m_bytes, bytes, size );
}
G4DAEBuffer::~G4DAEBuffer()
{
   delete m_bytes ;
}
char* G4DAEBuffer::GetBytes() 
{
   return m_bytes ; 
}
size_t G4DAEBuffer::GetSize() const
{
   return m_size ;
}
void G4DAEBuffer::Dump() const 
{
   printf("G4DAEBuffer::Dump size %lu 0x%lx \n", m_size, m_size );
   ::DumpBuffer( m_bytes, m_size );
}




