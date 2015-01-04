#include "G4DAEChroma/G4DAEBuffer.hh"
#include "G4DAEChroma/G4DAECommon.hh"

using namespace std ; 

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


G4DAEBuffer::G4DAEBuffer( size_t size, char* bytes ) : m_size(size), m_bytes(bytes)
{
   if(!size) return ;
   m_size = size ;
   m_bytes = new char[size] ;
   if(bytes) memcpy( m_bytes, bytes, size );
}


G4DAEBuffer::G4DAEBuffer( const char* path )
{
    FILE *fp = fopen(path, "r");
    if(!fp)
    {
        printf("G4DAEBuffer::G4DAEBuffer failed to open file %s \n", path );
        return ;  
    }

    if (fseek(fp, 0L, SEEK_END) == 0) 
    {
        long size = ftell(fp);
        if (size == -1)
        { 
            printf("G4DAEBuffer::G4DAEBuffer failed to determine file size %s \n", path );
            return ;  
        }
        if (fseek(fp, 0L, SEEK_SET) != 0) 
        {
            printf("G4DAEBuffer::G4DAEBuffer failed to seek to head  %s \n", path );
            return ;  
        }

        m_size = size ; 
        m_bytes = new char[size] ; 

        size_t nread = fread(m_bytes, sizeof(char), size, fp);
        if (nread == 0) 
        {
            printf("G4DAEBuffer::G4DAEBuffer failed to read  %s \n", path );
            return ;  
        }
        assert(nread == m_size);
    }
    fclose(fp);
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




