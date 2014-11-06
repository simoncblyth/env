#include "G4DAEChroma/G4DAEArray.hh"
#include "G4DAEChroma/G4DAECommon.hh"

#include "numpy.hpp"
#include <cassert>
#include <sys/stat.h> 

#include <sstream>
#include <iostream>
#include <iomanip>

using namespace std ; 


G4DAEArray::G4DAEArray( size_t itemcapacity, string itemshape, float* data) : 
             m_itemcapacity(itemcapacity), 
             m_itemcount(0), 
             m_buffer(NULL),
             m_buffersize(0)
{
    // itemshape such as "4,4" split to form vector and give itemsize of 4*4 
    isplit( m_itemshape, itemshape.c_str(), ',' );   
    assert( GetItemShapeString() == itemshape ); 
    m_itemsize = FormItemSize( m_itemshape, 0 );

    size_t nfloat = m_itemcapacity*m_itemsize ;

    m_data = new float[nfloat] ;
    if( data != NULL )   // copy floats into owned array 
    {
        m_itemcount = m_itemcapacity ; // when loading from float* data
        for(size_t n=0 ; n < nfloat ; ++n ) m_data[n] = data[n] ;   
    }
}

const char* G4DAEArray::GetBuffer() const
{
   return m_buffer ; 
}
size_t G4DAEArray::GetBufferSize() const
{
   return m_buffersize ;
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
   return m_itemcount*m_itemsize*sizeof(float) ;
}
size_t G4DAEArray::GetBytes() const
{
   return m_itemcapacity*m_itemsize*sizeof(float) ;
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
         << " buffersize: " << GetBufferSize() << " (bytes) "
         << " buffer: " << (void*)GetBuffer() 
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



void G4DAEArray::SaveToBuffer()
{
   bool fortran_order = false ; 
   string itemshape = GetItemShapeString();
   size_t expect_bytes = aoba::BufferSize<float>(m_itemcount, itemshape.c_str(), fortran_order  );  

   printf("G4DAEArray::SaveToBuffer itemcount %lu itemshape %s nbytes %zu \n", m_itemcount, itemshape.c_str(), expect_bytes );

   m_buffer = new char[expect_bytes];
   m_buffersize = expect_bytes ; 

   size_t wrote_bytes = aoba::BufferSaveArrayAsNumpy<float>( m_buffer, fortran_order, m_itemcount, itemshape.c_str(), m_data );  
   assert( wrote_bytes == expect_bytes );
   printf("G4DAEArray::SaveToBuffer wrote_bytes %zu \n", wrote_bytes );
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

   aoba::LoadArrayFromNumpy<float>(path, shape, data );

   size_t itemsize = FormItemSize( shape, 1);
   string itemshape = FormItemShapeString( shape, 1);
   size_t nitems = data.size()/itemsize ; 

   printf("G4DAEArray::Load [%s] itemsize %lu itemshape %s nitems %lu data.size %lu \n", 
       path.c_str(), itemsize, itemshape.c_str(), nitems, data.size() );

   //printf("G4DAEArray::Load DumpVector \n");
   //DumpVector( data, itemsize );

   //printf("G4DAEArray::Load DumpBuffer &data[0]\n");
   //DumpBuffer( reinterpret_cast<const char*>(&data[0]), data.size()*sizeof(float) );

   //printf("G4DAEArray::Load DumpBuffer data.data()\n");
   //DumpBuffer( reinterpret_cast<const char*>(data.data()), data.size()*sizeof(float) );


   G4DAEArray* arr = new G4DAEArray( nitems, itemshape, data.data() );  
   return arr ;
}

void G4DAEArray::DumpBuffer() const 
{
   printf("G4DAEArray::DumpBuffer arr->GetBufferSize()  %lu 0x%lx \n", GetBufferSize(), GetBufferSize() );
   ::DumpBuffer( GetBuffer(), GetBufferSize() );
}




G4DAEArray* G4DAEArray::LoadFromBuffer(const char* buffer, size_t buflen)
{
   printf("G4DAEArray::LoadFromBuffer [%zu][0x%lx]\n", buflen, buflen );
   ::DumpBuffer( buffer, buflen);

   std::vector<int>  shape ;
   std::vector<float> data ;

   aoba::BufferLoadArrayFromNumpy<float>(buffer, buflen, shape, data );

   size_t itemsize = FormItemSize( shape, 1);
   string itemshape = FormItemShapeString( shape, 1);
   size_t nitems = data.size()/itemsize ; 

   return new G4DAEArray( nitems, itemshape, data.data() );  
}




