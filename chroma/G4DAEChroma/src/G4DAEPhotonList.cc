#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAEArray.hh"

#include "numpy.hpp"
#include <cassert>
#include <sys/stat.h> 

#include <iostream>

using namespace std ; 


G4DAEPhotonList::G4DAEPhotonList( std::size_t itemcapacity, float* data) : m_array(NULL) 
{
    m_array = new G4DAEArray( itemcapacity, "4,4", data );
}

G4DAEPhotonList::G4DAEPhotonList( G4DAEArray* arr ) : m_array(arr) 
{
}
G4DAEPhotonList::G4DAEPhotonList( G4DAEPhotons* src ) : m_array(NULL)
{
    size_t itemcapacity = src->GetPhotonCount();
    m_array = new G4DAEArray( itemcapacity, "4,4", NULL );
    G4DAEPhotons::Transfer( this, src );
} 

G4DAEPhotonList* G4DAEPhotonList::Load(const char* evt, const char* key, const char* tmpl )
{
    G4DAEArray* array = G4DAEArray::Load(evt, key, tmpl);
    return new G4DAEPhotonList(array) ;  
}

G4DAEPhotonList* G4DAEPhotonList::LoadPath(const char* path, const char* key )
{
    G4DAEArray* array = G4DAEArray::LoadPath(path, key );
    return new G4DAEPhotonList(array) ;  
}

void G4DAEPhotonList::Save(const char* evt, const char* key, const char* tmpl )
{
    if(m_array) m_array->Save(evt, key, tmpl);
}
void G4DAEPhotonList::SavePath(const char* path, const char* key)
{
    if(m_array) m_array->SavePath(path, key );
}






string G4DAEPhotonList::GetPath(const char* evt, const char* tmpl )
{
    return G4DAEArray::GetPath(evt, tmpl);
}

G4DAEPhotonList::~G4DAEPhotonList()
{
   delete m_array ; 
}




// G4DAESerializable


G4DAEPhotonList*  G4DAEPhotonList::CreateOther(char* buffer, std::size_t buflen)
{
    if(!m_array) return NULL ;
    G4DAEArray* array = m_array->CreateOther(buffer, buflen);
    return new G4DAEPhotonList(array);
}

void G4DAEPhotonList::SaveToBuffer()
{
   if(!m_array) return ;
   m_array->SaveToBuffer();
}
void G4DAEPhotonList::DumpBuffer()
{
   if(!m_array) return ;
   m_array->DumpBuffer();
}
const char* G4DAEPhotonList::GetBufferBytes()
{
    return m_array ? m_array->GetBufferBytes() : NULL ;
}
std::size_t G4DAEPhotonList::GetBufferSize()
{
    return m_array ? m_array->GetBufferSize() : 0 ; 
}














// G4DAEPhotons


void G4DAEPhotonList::Print() const 
{
    if(m_array) m_array->Print();
}

void G4DAEPhotonList::Details(bool hit) const 
{
    cout <<  "G4DAEPhotonList::Details " << endl ;
    size_t count = GetPhotonCount();
    cout <<  "G4DAEPhotonList::Details [" << count << "]" << endl ;

    size_t index ;

    G4ThreeVector pos ;
    G4ThreeVector dir ;
    G4ThreeVector pol ;
    float _t ;
    float _wavelength ;
    int _pmtid ;

    for( index = 0 ; index < count ; index++ )
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



std::size_t G4DAEPhotonList::GetPhotonCount() const {
   return m_array ? m_array->GetSize() : 0 ;
}

std::string G4DAEPhotonList::GetPhotonDigest() const {
   return m_array ? m_array->GetDigest() : "" ;
}

void G4DAEPhotonList::ClearAllPhotons() {
   if(m_array) m_array->ClearAll();
}






void G4DAEPhotonList::GetPhoton( std::size_t index , G4ThreeVector& pos, G4ThreeVector& dir, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const
{
    float* data = m_array->GetItemPointer( index );
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
    float* data = m_array->GetNextPointer();

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



