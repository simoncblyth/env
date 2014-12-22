#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAECommon.hh"

#include <iostream>
using namespace std ;

G4DAEPhotonList::G4DAEPhotonList(G4DAEArray* array) : G4DAEList<G4DAEPhoton>(array) 
{
}

G4DAEPhotonList::G4DAEPhotonList( std::size_t itemcapacity, float* data) :  G4DAEList<G4DAEPhoton>(itemcapacity, data )
{
}

G4DAEPhotonList::~G4DAEPhotonList()
{
}


void G4DAEPhotonList::GetPhoton( std::size_t index , G4ThreeVector& pos, G4ThreeVector& dir, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const
{
    float* data = m_array->GetItemPointer( index );

    pos.setX(data[G4DAEPhoton::_post_x]);
    pos.setY(data[G4DAEPhoton::_post_y]);
    pos.setZ(data[G4DAEPhoton::_post_z]);
    _t = data[G4DAEPhoton::_post_w] ;

    dir.setX(data[G4DAEPhoton::_dirw_x]);
    dir.setY(data[G4DAEPhoton::_dirw_y]);
    dir.setZ(data[G4DAEPhoton::_dirw_z]);
    _wavelength = data[G4DAEPhoton::_dirw_w] ;

    pol.setX(data[G4DAEPhoton::_polw_x]);
    pol.setY(data[G4DAEPhoton::_polw_y]);
    pol.setZ(data[G4DAEPhoton::_polw_z]);
    //_weight = data[G4DAEPhoton::_polw_w];

    uif_t uifd[4] ; 
    uifd[0].f = data[G4DAEPhoton::_flag_x];
    uifd[1].f = data[G4DAEPhoton::_flag_y];
    uifd[2].f = data[G4DAEPhoton::_flag_z];
    uifd[3].f = data[G4DAEPhoton::_flag_w]; 

    // TODO: get this back to caller, struct to hold the quad ?
    int _photon_id ; 
    int _spare ; 
    unsigned int _flags ;

    _photon_id = uifd[0].i ;
    _spare     = uifd[1].i ;
    _flags     = uifd[2].u ;
    _pmtid     = uifd[3].i ;

}


void G4DAEPhotonList::AddPhoton( G4ThreeVector pos, G4ThreeVector dir, G4ThreeVector pol, float _t, float _wavelength, int _pmtid )
{
    // serialize photon into data structure

    float _weight = 1. ;
    float* data = m_array->GetNextPointer();

    data[G4DAEPhoton::_post_x] =  pos.x() ;
    data[G4DAEPhoton::_post_y] =  pos.y() ;
    data[G4DAEPhoton::_post_z] =  pos.z() ;
    data[G4DAEPhoton::_post_w] = _t ;

    data[G4DAEPhoton::_dirw_x] =  dir.x() ;
    data[G4DAEPhoton::_dirw_y] =  dir.y() ;
    data[G4DAEPhoton::_dirw_z] =  dir.z() ;
    data[G4DAEPhoton::_dirw_w] = _wavelength ;

    data[G4DAEPhoton::_polw_x] =  pol.x() ;
    data[G4DAEPhoton::_polw_y] =  pol.y() ;
    data[G4DAEPhoton::_polw_z] =  pol.z() ;
    data[G4DAEPhoton::_polw_w] = _weight ;

    int _photon_id = 0; 
    int _spare     = 0; 
    unsigned int _flags     = 0 ;

    uif_t uifd[4] ; 
    uifd[0].i = _photon_id ;
    uifd[1].i = _spare ;
    uifd[2].u = _flags     ;
    uifd[3].i = _pmtid     ; 

    data[G4DAEPhoton::_flag_x] =  uifd[0].f ;
    data[G4DAEPhoton::_flag_y] =  uifd[1].f ;
    data[G4DAEPhoton::_flag_z] =  uifd[2].f ;
    data[G4DAEPhoton::_flag_w] =  uifd[3].f ;

}


void G4DAEPhotonList::Details(bool /*hit*/) const 
{
    cout <<  "G4DAEPhotonList::Details " << endl ;
    size_t count = GetCount();
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

void G4DAEPhotonList::Print(const char* msg) const 
{
    if(m_array) m_array->Print(msg);
    //if(m_link) m_link->Print(msg);
}


