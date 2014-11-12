#include "G4DAEChroma/G4DAEArray.hh"
#include "G4DAEChroma/G4DAEBuffer.hh"
#include "G4DAEChroma/G4DAECommon.hh"

#include "numpy.hpp"
#include <cassert>
#include <sys/stat.h> 

#include <sstream>
#include <iostream>
#include <iomanip>

using namespace std ; 

G4DAEArray* G4DAEArray::CreateOther(char* bytes, size_t size)
{
   // used by zombies
   return new G4DAEArray(bytes, size);
}

G4DAEArray::G4DAEArray(char* bytes, size_t size)
{
    Zero();
    Populate(bytes, size);
}

G4DAEArray::G4DAEArray( size_t itemcapacity, string itemshape, float* data) 
{
    Zero();
    Populate( itemcapacity, itemshape, data );
}

void G4DAEArray::Zero()
{
    m_data = NULL ;
    m_buffer = NULL ;
    m_itemcapacity = 0 ; 
    m_itemcount  = 0 ; 
}

void G4DAEArray::ClearAll()
{
    delete [] m_data;
    delete m_buffer ; 
    Zero();
}


void G4DAEArray::Populate( char* bytes, size_t size )
{
    if(!bytes) return;  // zombie expedient, for zombie->Create(bytes, size) 

#ifdef VERBOSE
    printf("G4DAEArray::Populate [%zu][0x%lx] ::DumpBuffer \n", size, size );
    ::DumpBuffer( bytes, size);
#endif

    std::vector<int>  shape ;
    std::vector<float> data ;

    aoba::BufferLoadArrayFromNumpy<float>(bytes, size, shape, data );

    string itemshape = FormItemShapeString( shape, 1);
    size_t itemsize = FormItemSize( shape, 1);
    size_t nitems = data.size()/itemsize ; 

    Populate( nitems, itemshape, data.data() );
}


void G4DAEArray::Populate( size_t nitems, string itemshape, float* data )
{

    if(!nitems) return;  // zombie expedient, for zombie->Create(bytes, size) 

    m_itemcapacity = nitems ; 

    // itemshape such as "4,4" split to form vector and give itemsize of 4*4 
    isplit( m_itemshape, itemshape.c_str(), ',' );   assert( GetItemShapeString() == itemshape ); 

    m_itemsize = FormItemSize( m_itemshape, 0 );

    size_t n = nitems*m_itemsize ;
    m_data = new float[n] ;
    m_buffer = NULL ; 

    if(data)   // copy floats into owned array 
    {
        m_itemcount = m_itemcapacity ; 
        while(n--) m_data[n] = data[n] ;   
    }
    else
    {
        m_itemcount = 0 ;
    }

}


float* G4DAEArray::GetItemPointer(std::size_t index)
{
    assert(index < m_itemcapacity );
    float* data = m_data + index*m_itemsize ;   
    return data ; 
}

float* G4DAEArray::GetNextPointer()
{
    // hmm need capability to grow the buffer for real collection 
    assert(m_itemcount < m_itemcapacity );

    //cout << "G4DAEArray::GetNextPointer itemcount " << m_itemcount << " itemsize " << m_itemsize << endl ; 

    float* data = m_data + m_itemcount*m_itemsize ;   
    m_itemcount++ ; 

    return data ; 
}







G4DAEBuffer* G4DAEArray::GetBuffer() const
{
   return m_buffer ; 
}


G4DAEArray::~G4DAEArray()
{
   delete[] m_data ; 
   delete m_buffer ;
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

void G4DAEArray::Print(const char* msg ) const 
{
    cout << msg 
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
   SavePath(path.c_str());
}

void G4DAEArray::SavePath(const char* _path, const char* /*key*/)
{
   string path(_path);
   string itemshape = GetItemShapeString();
   aoba::SaveArrayAsNumpy<float>(path, m_itemcount, itemshape.c_str(), m_data );
#ifdef VERBOSE
   printf("G4DAEArray::SavePath [%s] itemcount %lu itemshape %s \n", path.c_str(), m_itemcount, itemshape.c_str() );
#endif
}


// Serializable protocol methods

void G4DAEArray::SaveToBuffer()
{
   bool fortran_order = false ; 
   string itemshape = GetItemShapeString();
   size_t nbytes = aoba::BufferSize<float>(m_itemcount, itemshape.c_str(), fortran_order  );  

   delete m_buffer ; 
   m_buffer = new G4DAEBuffer(nbytes); 

   size_t wbytes = aoba::BufferSaveArrayAsNumpy<float>( m_buffer->GetBytes(), fortran_order, m_itemcount, itemshape.c_str(), m_data );  
   assert( wbytes == nbytes );

#ifdef VERBOSE
   printf("G4DAEArray::SaveToBuffer itemcount %lu itemshape %s nbytes %zu wrote bytes %zu \n", m_itemcount, itemshape.c_str(), nbytes, wbytes );
#endif
}

const char* G4DAEArray::GetBufferBytes()
{
   return m_buffer->GetBytes();
}
std::size_t G4DAEArray::GetBufferSize()
{
   return m_buffer->GetSize();
}
void G4DAEArray::DumpBuffer()
{
   return m_buffer->Dump();
}



G4DAEArray* G4DAEArray::Load(const char* evt, const char* key, const char* tmpl )
{
   string path = GetPath(evt, tmpl);
   if( path.empty() ) 
   {
      printf("G4DAEArray::Load : failed to format path from tmpl  %s and evt %s \n", tmpl, evt );  
      return NULL ; 
   }
   return LoadPath( path.c_str(), key);
}

G4DAEArray* G4DAEArray::LoadPath(const char* _path, const char* /*key*/ )
{
   string path(_path);
   std::vector<int>  shape ;
   std::vector<float> data ;

   aoba::LoadArrayFromNumpy<float>(path, shape, data );

   string itemshape = FormItemShapeString( shape, 1);
   size_t itemsize = FormItemSize( shape, 1);
   size_t nitems = data.size()/itemsize ; 

#ifdef VERBOSE
   printf("G4DAEArray::Load [%s] itemsize %lu itemshape %s nitems %lu data.size %lu \n", 
       path.c_str(), itemsize, itemshape.c_str(), nitems, data.size() );
#endif

   return new G4DAEArray( nitems, itemshape, data.data() );  
}





