#include "G4DAEChroma/G4DAEArray.hh"
#include "G4DAEChroma/G4DAEBuffer.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/numpy.hpp"

#include <cassert>
#include <sys/stat.h> 
#include <libgen.h> 

#include <sstream>
#include <iostream>
#include <iomanip>

using namespace std ; 

const char* G4DAEArray::MAGIC = "\x93NUMPY" ; 
const size_t G4DAEArray::INITCAPACITY = 10000 ; 
const float G4DAEArray::GROWTH = 1.5 ; 

G4DAEArray* G4DAEArray::CreateOther(char* bytes, size_t size)
{
   // used by zombies
   return new G4DAEArray(bytes, size);
}

G4DAEArray::G4DAEArray(char* bytes, size_t size, float growth) 
          : 
          m_initcapacity(INITCAPACITY),
          m_growthfactor(growth)
{
    Zero();
    Populate(bytes, size);
}

G4DAEArray::G4DAEArray( size_t initcapacity, string itemshape, float* data, float growth ) 
          : 
          m_initcapacity(initcapacity),
          m_growthfactor(growth)
{
    Zero();
    Populate( initcapacity, itemshape, data );
}


G4DAEArray* G4DAEArray::Slice( int start, int stop, int step )
{
    return new G4DAEArray( this, start, stop, step );
}

G4DAEArray::G4DAEArray( G4DAEArray* src, int start, int stop, int step )
{

    size_t nsrc = src->GetSize();
    printf("G4DAEArray::G4DAEArray sliced copy ctor : nsrc %zu start/stop/step %d/%d/%d \n", nsrc, start, stop, step );

    if(start == INT_MAX) start = 0 ;
    if(stop  == INT_MAX) stop  = nsrc ;
    if(step  == INT_MAX) step  = 1 ;

    if( start < 0) start = nsrc + start ;
    if( stop < 0)  stop  = nsrc + stop  ;

    if(start < 0 || stop < 0)
    {
        printf("G4DAEArray::Transfer unexpected nsrc/start/stop/step %zu %d %d %d \n", nsrc,start, stop, step);
        return ;  
    }

    m_initcapacity = abs( stop - start )/abs(step) ; // initial guess, but the array will grow as needed anyhow
    m_growthfactor = src->GetGrowthFactor();

    Zero();

    InitializeItemShape(src->GetItemShapeString());

    G4DAEArray::Transfer( this, src, start, stop, step );
} 

void G4DAEArray::Transfer( G4DAEArray* dest , G4DAEArray* src, size_t start, size_t stop, int step )
{
    size_t nsrc = src->GetSize();
    assert(src->GetItemSize() == dest->GetItemSize());

    size_t nbytes = src->GetItemSize()*1*sizeof(float); 
    size_t take=0; 

    //TODO: test behaviour with non-existing start/stop
    for(size_t index=start ; index < stop ; index+=step)
    {
        float* s = src->GetItemPointer(index);
        if(!s)
        {
            printf("G4DAEArray::Transfer  NULL item \n");
            return ; 
        } 
        float* d = dest->GetNextPointer();
        memcpy( d, s, nbytes );   
        take++;
    }

    // itemwise copying, could be made much more efficient 
    // by copying multiple contiguous items at once depending on start, stop, step 
   
    printf("G4DAEArray::Transfer nsrc %zu nbytes %zu start/stop/step %zu/%zu/%d took %zu  \n", nsrc,nbytes, start,stop,step, take);

}




void G4DAEArray::Zero()
{
    m_data = NULL ;
    m_buffer = NULL ;
    m_itemcount  = 0 ; 
    m_itemcapacity = 0 ; 
}

void G4DAEArray::ClearAll()
{
    free(m_data);
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
    

    // interpreting (bytes, size)  as serialized NPY array
    std::vector<int>  shape ;
    std::vector<float> data ;
    aoba::BufferLoadArrayFromNumpy<float>(bytes, size, shape, data );

    size_t from = 1 ;   //  first dimension excluded
    string itemshape = FormItemShapeString( shape, from);
    size_t itemsize = FormItemSize( shape, from);
    size_t nitems = data.size()/itemsize ; 

    // shovelling bytes into native float* array m_data
    Populate( nitems, itemshape, data.data() );
}


void G4DAEArray::Allocate( size_t nitems )
{
    size_t nfloat = nitems*m_itemsize ;
    //printf("G4DAEArray::Allocate nitems %zu nfloat %zu \n", nitems, nfloat );
    m_data = (float*)malloc( nfloat*sizeof(float) ) ;
    m_itemcapacity = nitems ; 
    m_buffer = NULL ;   
}

void G4DAEArray::Extend(size_t nitems )
{
   size_t nfloat = nitems*m_itemsize ;
   float* tmp = (float*)realloc(m_data, nfloat*sizeof(float)  );
   if(tmp) 
   {
       printf("G4DAEArray::Extend to nitems %zu nfloat  %zu \n", nitems, nfloat );
       m_data = tmp;
       m_itemcapacity = nitems ; 
   }
   else 
   {
       printf("G4DAEArray::Extend FAILURE nitems %zu nfloat %zu \n", nitems, nfloat );
   }
}

float* G4DAEArray::GetNextPointer()
{
    if(m_itemcount == m_itemcapacity)
    {
         if(m_itemcapacity == 0)  // following a ClearAll  need to make new allocation
         {
             printf("G4DAEArray::GetNextPointer allocating to initcapacity %zu following a ClearAll ", m_initcapacity );
             Allocate(m_initcapacity); // allocates and bumps up itemcapacity 
         }
         else
         {
             Extend(m_itemcapacity*m_growthfactor);
         }
    }
    assert(m_itemcount < m_itemcapacity );

    float* data = m_data + m_itemcount*m_itemsize ;   

    m_itemcount++ ; 
    return data ; 
}

void G4DAEArray::InitializeItemShape( string itemshape )
{
    isplit( m_itemshape, itemshape.c_str(), ',' );   // populate m_itemshape   vector<int> 

    assert( GetItemShapeString() == itemshape ); 

    m_itemsize = FormItemSize( m_itemshape, 0 );     // eg 16 for "4,4" (1st dim already dropped)

}


void G4DAEArray::Populate( size_t nitems, string itemshape, float* data )
{  
    /*
       #. nitems eg 1000 for full shape 1000,4,4
       #. itemshape with 1st dim already dropped eg  "4,4" 


    Sets member variables characterizing the data and copies
    it into m_data 
    */

    if(!nitems) return;  // zombie expedient, for zombie->Create(bytes, size) 

    InitializeItemShape(itemshape);

    Allocate(nitems);

    if(data)   
    {
        m_itemcount = nitems ; 
        size_t nbytes = m_itemsize*nitems*sizeof(float); 
        memcpy( m_data, data, nbytes );
    }
    else
    {
        m_itemcount = 0 ;
    }
}



float* G4DAEArray::GetItemPointer(std::size_t index)
{
   // only gets existing items 
    return (index < m_itemcount) ?  m_data + index*m_itemsize : NULL  ;   
}



G4DAEArray::~G4DAEArray()
{
   free(m_data) ; 
   delete m_buffer ;
}


string G4DAEArray::GetItemShapeString() const 
{
    // full shape string eg "1000,4,4"
    return FormItemShapeString(m_itemshape, 0);
}

size_t G4DAEArray::FormItemSize(const vector<int>& itemshape, size_t from) 
{
    //
    //  eg for itemshape 1000,4,4  (1000 4x4 matrices) 
    //     using from = 1 gives ItemSize  16 
    //
    size_t itemsize = 1 ; 
    for(size_t d=from ; d<itemshape.size(); ++d) itemsize *= itemshape[d]; 
    return itemsize ; 
}

string G4DAEArray::FormItemShapeString(const vector<int>& itemshape, size_t from) 
{
    //
    //  eg for itemshape 1000,4,4  (1000 4x4 matrices) 
    //     using from = 1 gives ItemShapeString "4,4"
    //
    stringstream ss ; 
    size_t nidim = itemshape.size() ; 
    for(size_t d=from ; d<nidim ; ++d)
    {
        ss << itemshape[d] ;
        if( d < nidim -1 ) ss << "," ; 
    }
    return ss.str();
}


float G4DAEArray::GetGrowthFactor() const
{
    return m_growthfactor ; 
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
         << " size: "     << setw(4) << GetSize() 
         << " capacity: " << setw(4) << GetCapacity() 
         << " itemsize: " << setw(4) << GetItemSize() 
         << " itemshape: " << setw(4) << GetItemShapeString() 
         << " bytesused: " << setw(7) << GetBytesUsed() 
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

void G4DAEArray::Save(const char* evt, const char* /*evtkey*/, const char* tmpl)
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
   printf("G4DAEArray::SavePath [%s] itemcount %lu itemshape %s \n", path.c_str(), m_itemcount, itemshape.c_str() );

   char* dirp = dirname((char*)_path);
   mkdirp(dirp, 0777);
   //printf("G4DAEArray::SavePath dirp [%s] mkdirp rc %d \n", dirp, rc ); 

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

   // pre-calculate total buffer size including the padded header
   size_t nbytes = aoba::BufferSize<float>(m_itemcount, itemshape.c_str(), fortran_order  );  

   delete m_buffer ; 
   m_buffer = new G4DAEBuffer(nbytes); 

   size_t wbytes = aoba::BufferSaveArrayAsNumpy<float>( m_buffer->GetBytes(), fortran_order, m_itemcount, itemshape.c_str(), m_data );  
   assert( wbytes == nbytes );

#ifdef VERBOSE
   printf("G4DAEArray::SaveToBuffer itemcount %lu itemshape %s nbytes %zu wrote bytes %zu \n", m_itemcount, itemshape.c_str(), nbytes, wbytes );
#endif
}





//  buffer access for serialization

G4DAEBuffer* G4DAEArray::GetBuffer() const
{
   return m_buffer ; 
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
   if(!m_buffer)
   {
       printf("G4DAEArray::DumpBuffer buffer is NULL \n");
       return;
   }
   m_buffer->Dump();
}
const char* G4DAEArray::GetMagic()
{
   return MAGIC ;  
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





