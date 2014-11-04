#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAECommon.hh"

#include "numpy.hpp"
#include <cassert>
#include <sys/stat.h> 

#include <iostream>

using namespace std ; 


G4DAEPhotonList::G4DAEPhotonList( std::size_t itemcapacity, float* data) : 
             m_itemcapacity(itemcapacity), 
             m_itemsize(4*4), 
             m_itemcount(0)
{
    m_data = new float[m_itemcapacity*m_itemsize] ;
    if( data != NULL )   // copy the buffer
    {
        const char* source = reinterpret_cast<const char*>(data);
        char* dest = reinterpret_cast<char*>(m_data) ;
        size_t nbytes = m_itemcapacity*m_itemsize*sizeof(float);
        memcpy( dest, source, nbytes ) ; 
        m_itemcount = m_itemcapacity ; // when loading from buffers
    }
}


G4DAEPhotonList::~G4DAEPhotonList()
{
    if(m_data) delete[] m_data ; 
}


size_t G4DAEPhotonList::GetSize() const
{
   return m_itemcount ;
}
size_t G4DAEPhotonList::GetBytesUsed() const
{
   return m_itemcount*m_itemsize ;
}
size_t G4DAEPhotonList::GetItemSize() const
{
   return m_itemsize ;
}
size_t G4DAEPhotonList::GetCapacity() const
{
   return m_itemcapacity ;
}
string G4DAEPhotonList::GetDigest() const
{
    const char* data = reinterpret_cast<const char*>(m_data);
    size_t nbytes = m_itemcount*m_itemsize ;
    return md5digest( data, nbytes ); 
} 
void G4DAEPhotonList::Print() const 
{
    cout <<  "G4DAEPhotonList::Print " 
         << " size: " << GetSize() 
         << " capacity: " << GetCapacity() 
         << " itemsize: " << GetItemSize() 
         << " bytesused: " << GetBytesUsed() 
         << " digest: " << GetDigest() 
         << endl ;    
} 
void G4DAEPhotonList::Details(bool hit) const 
{
    cout <<  "G4DAEPhotonList::Details [" << GetSize() << "]" << endl ;

    size_t index ;

    G4ThreeVector pos ;
    G4ThreeVector dir ;
    G4ThreeVector pol ;
    float _t ;
    float _wavelength ;
    int _pmtid ;

    for( index = 0 ; index < GetSize() ; index++ )
    {
        GetPhoton( index , pos, dir, pol, _t, _wavelength, _pmtid );
        cout << " index " << index
             << " pos " << pos
             << " dir " << dir
             << " pol " << pol
             << " _t " << _t
             << " _wavelength " << _wavelength
             << " _pmtid " << (void*)_pmtid
             << endl ;
    }
}


enum {
   post_x,
   post_y,
   post_z,
   post_w,

   dirw_x,
   dirw_y,
   dirw_z,
   dirw_w,

   polw_x,
   polw_y,
   polw_z,
   polw_w,

   flag_x,
   flag_y,
   flag_z,
   flag_w
};


// using union for writing ints into float slot 
typedef union {
    float f ;
    int i ;
} uif_t ;  


void G4DAEPhotonList::GetPhoton( std::size_t index , G4ThreeVector& pos, G4ThreeVector& dir, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const
{
    assert(index < m_itemcapacity );
    float* data = m_data + index*m_itemsize ;   

    pos.setX(data[post_x]);
    pos.setY(data[post_y]);
    pos.setZ(data[post_z]);
    _t = data[post_w] ;

    dir.setX(data[dirw_x]);
    dir.setY(data[dirw_y]);
    dir.setZ(data[dirw_z]);
    _wavelength = data[dirw_w] ;

    pol.setX(data[polw_x]);
    pol.setY(data[polw_y]);
    pol.setZ(data[polw_z]);
    //_weight = data[polw_w];

    uif_t uifd[4] ; 
    uifd[0].f = data[flag_x];
    uifd[1].f = data[flag_y];
    uifd[2].f = data[flag_z];
    uifd[3].f = data[flag_w]; 

    // =  uifd[0].i ;
    // =  uifd[1].i ;
    // =  uifd[2].i ;
    _pmtid =  uifd[3].i ;

}


void G4DAEPhotonList::AddPhoton( G4ThreeVector pos, G4ThreeVector dir, G4ThreeVector pol, float _t, float _wavelength, int _pmtid )
{
    // serialize photon into data structure

    float _weight = 1. ;

    assert(m_itemcount < m_itemcapacity );

    cout << "G4DAEPhotonList::AddPhoton itemcount " << m_itemcount << " itemsize " << m_itemsize << endl ; 

    float* data = m_data + m_itemcount*m_itemsize ;   
    m_itemcount++ ; 

    data[post_x] =  pos.x() ;
    data[post_y] =  pos.y() ;
    data[post_z] =  pos.z() ;
    data[post_w] = _t ;

    data[dirw_x] =  dir.x() ;
    data[dirw_y] =  dir.y() ;
    data[dirw_z] =  dir.z() ;
    data[dirw_w] = _wavelength ;

    data[polw_x] =  pol.x() ;
    data[polw_y] =  pol.y() ;
    data[polw_z] =  pol.z() ;
    data[polw_w] = _weight ;

    //assert(sizeof(float) == sizeof(int)); 

    uif_t uifd[4] ; 
    uifd[0].i = 10 ;
    uifd[1].i = 20 ;
    uifd[2].i = 30 ;
    uifd[3].i = _pmtid ; 

    data[flag_x] =  uifd[0].f ;
    data[flag_y] =  uifd[1].f ;
    data[flag_z] =  uifd[2].f ;
    data[flag_w] =  uifd[3].f ;

}


string G4DAEPhotonList::GetPath( const char* evt , const char* tmpl )
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



void G4DAEPhotonList::Save(const char* evt, const char* evtkey, const char* tmpl)
{
   string path = GetPath(evt, tmpl);
   if( path.empty() )
   {   
      printf("G4DAEPhotonList::Save : failed to format path from tmpl  %s and evt %s \n", tmpl, evt );  
      return; 
   }   

   bool fortran_order = false ; 
   const int ndim = 3 ; 
   const int shape[] = {m_itemcount, 4, 4} ;

   printf("G4DAEPhotonList::Save [%s]\n", path.c_str() );
   aoba::SaveArrayAsNumpy<float>(path, fortran_order, ndim, shape, m_data );

}



G4DAEPhotonList* G4DAEPhotonList::Load(const char* evt, const char* key, const char* tmpl )
{
    string path = GetPath(evt, tmpl);
    if( path.empty() )
    {
      printf("G4DAEPhotonList::Load : failed to format path from tmpl  %s and evt %s \n", tmpl, evt );  
      return NULL ; 
    }


   std::vector<int>  shape ;
   std::vector<float> data ;

   printf("G4DAEPhotonList::Load [%s]\n", path.c_str() );
   aoba::LoadArrayFromNumpy<float>(path, shape, data );

   assert(shape.size() == 3);
   assert(shape[1] == 4 && shape[2] == 4);

   size_t vsize = data.size();
   size_t itemsize = shape[1]*shape[2] ; 
   size_t nitems = vsize / itemsize ;  

   /*
   cout << "vsize " << vsize << endl ; 
   cout << "itemsize " << itemsize << endl ; 
   cout << "nitems " << nitems << endl ; 
   */

   return new G4DAEPhotonList(nitems, data.data() );  
}




