#include "G4DAEChroma/G4DAEPhotonListOld.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAEArray.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"
#include "G4DAEChroma/numpy.hpp"

#include <cassert>
#include <sys/stat.h> 

#include <iostream>

using namespace std ; 

const char* G4DAEPhotonListOld::TMPL = "DAE_PATH_TEMPLATE_NPY" ;
const char* G4DAEPhotonListOld::SHAPE = "4,4" ;
const char* G4DAEPhotonListOld::KEY   = "NPY" ;


G4DAEPhotonListOld::G4DAEPhotonListOld( std::size_t itemcapacity, float* data) : m_array(NULL), m_link(NULL) 
{
    m_array = new G4DAEArray( itemcapacity, SHAPE, data );
}

G4DAEPhotonListOld::G4DAEPhotonListOld( G4DAEArray* arr ) : m_array(arr), m_link(NULL) 
{
}
G4DAEPhotonListOld::G4DAEPhotonListOld( G4DAEPhotons* src, int a, int b ) : m_array(NULL), m_link(NULL)
{
    // provides conversions between different implementation of photon lists 
    size_t itemcapacity = b - a > 0 ? b - a : src->GetCount();
    m_array = new G4DAEArray( itemcapacity, SHAPE, NULL );
    G4DAEPhotons::Transfer( this, src, a, b );
} 

G4DAEPhotonListOld* G4DAEPhotonListOld::Slice( int a, int b )
{
    return new G4DAEPhotonListOld( this, a, b );
}


G4DAEPhotonListOld::~G4DAEPhotonListOld()
{
   delete m_array ; 
   // delete m_link ; **NOT DELETING LINK : REGARDED AS WEAK REFERENCE**
}



G4DAEPhotonListOld* G4DAEPhotonListOld::Load(const char* evt, const char* key, const char* tmpl )
{
    return G4DAEArray::Load<G4DAEPhotonListOld>(evt, key, tmpl);
}

G4DAEPhotonListOld* G4DAEPhotonListOld::LoadPath(const char* path, const char* key )
{
    return G4DAEArray::LoadPath<G4DAEPhotonListOld>(path, key );
}

void G4DAEPhotonListOld::Save(const char* evt, const char* key, const char* tmpl )
{
    if(m_array) m_array->Save(evt, key, tmpl);
}
void G4DAEPhotonListOld::SavePath(const char* path, const char* key)
{
    if(m_array) m_array->SavePath(path, key );
}






string G4DAEPhotonListOld::GetPath(const char* evt, const char* tmpl )
{
    return G4DAEArray::GetPath(evt, tmpl);
}



// hmm duplicating G4DAEArrayHolder

std::size_t G4DAEPhotonListOld::GetCount() const {
    return m_array ? m_array->GetSize() : 0 ;
}

std::string G4DAEPhotonListOld::GetDigest() const {
    return m_array ? m_array->GetDigest() : "" ;
}

void G4DAEPhotonListOld::ClearAll() {
    if(m_array) m_array->ClearAll();
}



// G4DAESerializable


G4DAEPhotonListOld*  G4DAEPhotonListOld::CreateOther(char* buffer, std::size_t buflen)
{
    if(!m_array) return NULL ;
    G4DAEArray* array = m_array->CreateOther(buffer, buflen);
    return new G4DAEPhotonListOld(array);
}

void G4DAEPhotonListOld::SaveToBuffer()
{
   if(!m_array) return ;
   m_array->SaveToBuffer();
}
void G4DAEPhotonListOld::DumpBuffer()
{
   if(!m_array) return ;
   m_array->DumpBuffer();
}
const char* G4DAEPhotonListOld::GetBufferBytes()
{
    return m_array ? m_array->GetBufferBytes() : NULL ;
}
std::size_t G4DAEPhotonListOld::GetBufferSize()
{
    return m_array ? m_array->GetBufferSize() : 0 ; 
}
const char* G4DAEPhotonListOld::GetMagic()
{
    return m_array ? m_array->GetMagic() : NULL ;
}




void G4DAEPhotonListOld::SetLink(G4DAEMetadata* link )
{
    m_link = link ;
}
G4DAEMetadata* G4DAEPhotonListOld::GetLink()
{
    return m_link ;
}














// G4DAEPhotons


void G4DAEPhotonListOld::Print(const char* msg) const 
{
    if(m_array) m_array->Print(msg);
    //if(m_link) m_link->Print(msg);
}

void G4DAEPhotonListOld::Details(bool /*hit*/) const 
{
    cout <<  "G4DAEPhotonListOld::Details " << endl ;
    size_t count = GetCount();
    cout <<  "G4DAEPhotonListOld::Details [" << count << "]" << endl ;

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









void G4DAEPhotonListOld::GetPhoton( std::size_t index , G4ThreeVector& pos, G4ThreeVector& dir, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const
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

    // TODO: get this back to caller, struct to hold the quad ?
    int _photon_id ; 
    int _spare ; 
    unsigned int _flags ;

    _photon_id = uifd[0].i ;
    _spare     = uifd[1].i ;
    _flags     = uifd[2].u ;
    _pmtid     = uifd[3].i ;

}


void G4DAEPhotonListOld::AddPhoton( G4ThreeVector pos, G4ThreeVector dir, G4ThreeVector pol, float _t, float _wavelength, int _pmtid )
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

    int _photon_id = 0; 
    int _spare     = 0; 
    unsigned int _flags     = 0 ;

    uif_t uifd[4] ; 
    uifd[0].i = _photon_id ;
    uifd[1].i = _spare ;
    uifd[2].u = _flags     ;
    uifd[3].i = _pmtid     ; 

    data[flag_x] =  uifd[0].f ;
    data[flag_y] =  uifd[1].f ;
    data[flag_z] =  uifd[2].f ;
    data[flag_w] =  uifd[3].f ;


}




