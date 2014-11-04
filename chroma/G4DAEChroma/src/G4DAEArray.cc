#include "G4DAEChroma/G4DAEArray.hh"
#include "G4DAEChroma/G4DAECommon.hh"

#include "numpy.hpp"
#include <cassert>
#include <sys/stat.h> 

#include <sstream>
#include <iostream>

using namespace std ; 


G4DAEArray::G4DAEArray( size_t itemcapacity, string itemshape, float* data) : 
             m_itemcapacity(itemcapacity), 
             m_itemcount(0)
{
    // itemshape such as "4,4" split to form vector and give itemsize of 4*4 
    isplit( m_itemshape, itemshape.c_str(), ',' );   
    assert( GetItemShapeString() == itemshape ); 
    m_itemsize = FormItemSize( m_itemshape, 0 );

    m_data = new float[m_itemcapacity*m_itemsize] ;
    if( data != NULL )   // copy the buffer
    {
        m_itemcount = m_itemcapacity ; // when loading from buffers
        const char* source = GetBuffer();
        size_t nbytes = GetBufferSize();
        char* dest = reinterpret_cast<char*>(m_data) ;
        memcpy( dest, source, nbytes ) ; 
    }
}

const char* G4DAEArray::GetBuffer() const
{
   return reinterpret_cast<const char*>(m_data) ; 
}
size_t G4DAEArray::GetBufferSize() const
{
   return m_itemcount*m_itemsize*sizeof(float);
}



G4DAEArray::~G4DAEArray()
{
    if(m_data) delete[] m_data ; 
}


string G4DAEArray::GetItemShapeString() const 
{
    return FormItemShapeString(m_itemshape, 0);
}

size_t G4DAEArray::FormItemSize(const vector<int>& itemshape, int from) 
{
    size_t itemsize = 1 ; 
    for(int d=from ; d<itemshape.size(); ++d) itemsize *= itemshape[d]; 
    return itemsize ; 
}

string G4DAEArray::FormItemShapeString(const vector<int>& itemshape, int from) 
{
    stringstream ss ; 
    size_t nidim = itemshape.size() ; 
    for(int d=from ; d<nidim ; ++d)
    {
        ss << itemshape[d] ;
        if( d < nidim -1 ) ss << "," ; 
    }
    return ss.str();
}


size_t G4DAEArray::GetItemSize() const
{
   return m_itemsize ;
}
size_t G4DAEArray::GetSize() const
{
   return m_itemcount ;
}
size_t G4DAEArray::GetBytesUsed() const
{
   return m_itemcount*m_itemsize ;
}
size_t G4DAEArray::GetBytes() const
{
   return m_itemcapacity*m_itemsize ;
}
size_t G4DAEArray::GetCapacity() const
{
   return m_itemcapacity ;
}
string G4DAEArray::GetDigest() const
{
    const char* data = reinterpret_cast<const char*>(m_data);
    size_t nbytes = m_itemcount*m_itemsize ;
    return md5digest( data, nbytes ); 
} 

void G4DAEArray::Print() const 
{
    cout <<  "G4DAEArray::Print " 
         << " size: " << GetSize() 
         << " capacity: " << GetCapacity() 
         << " itemsize: " << GetItemSize() 
         << " itemshape: " << GetItemShapeString() 
         << " bytesused: " << GetBytesUsed() 
         << " digest: " << GetDigest() 
         << endl ;    
} 



string G4DAEArray::GetPath( const char* evt , const char* tmpl )
{
   string empty ;
   const char* evtfmt  = getenv(tmpl);
   if(evtfmt == NULL ){
      printf("tmpl %s : missing : use \"export-;export-export\" to define  \n", tmpl );
      return empty; 
   }   
   char evtpath[256];
   if (sprintf(evtpath, evtfmt, evt ) < 0) return empty;
   return string( evtpath );
}


void G4DAEArray::Save(const char* evt, const char* evtkey, const char* tmpl)
{
   string path = GetPath(evt, tmpl);
   if( path.empty() )
   {   
      printf("G4DAEArray::Save : failed to format path from tmpl  %s and evt %s \n", tmpl, evt );  
      return; 
   }   
   string itemshape = GetItemShapeString();
   printf("G4DAEArray::Save [%s] itemcount %lu itemshape %s \n", path.c_str(), m_itemcount, itemshape.c_str() );
   aoba::SaveArrayAsNumpy<float>(path, m_itemcount, itemshape.c_str(), m_data );
}



G4DAEArray* G4DAEArray::Load(const char* evt, const char* key, const char* tmpl )
{
   string path = GetPath(evt, tmpl);
   if( path.empty() )
   {
      printf("G4DAEArray::Load : failed to format path from tmpl  %s and evt %s \n", tmpl, evt );  
      return NULL ; 
   }

   std::vector<int>  shape ;
   std::vector<float> data ;

   printf("G4DAEArray::Load [%s]\n", path.c_str() );
   aoba::LoadArrayFromNumpy<float>(path, shape, data );

   size_t itemsize = FormItemSize( shape, 1);
   string itemshape = FormItemShapeString( shape, 1);
   size_t nitems = data.size()/itemsize ; 

   return new G4DAEArray( nitems, itemshape, data.data() );  
}

G4DAEArray* G4DAEArray::LoadFromBuffer(const char* buffer, size_t buflen)
{
   printf("G4DAEArray::LoadFromBuffer [%zu]\n", buflen );

   std::vector<int>  shape ;
   std::vector<float> data ;

   aoba::BufferLoadArrayFromNumpy<float>(buffer, buflen, shape, data );

   size_t itemsize = FormItemSize( shape, 1);
   string itemshape = FormItemShapeString( shape, 1);
   size_t nitems = data.size()/itemsize ; 

   return new G4DAEArray( nitems, itemshape, data.data() );  
}




