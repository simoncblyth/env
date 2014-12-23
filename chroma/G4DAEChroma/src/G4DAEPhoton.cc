#include "G4DAEChroma/G4DAEPhoton.hh"
#include "G4DAEChroma/G4DAEArrayHolder.hh" 
#include "G4DAEChroma/G4DAECommon.hh" 

#include "G4Track.hh"
#include "G4VProcess.hh"

#include <iostream>
using namespace std ;

const char* G4DAEPhoton::TMPL = "DAE_PHOTON_PATH_TEMPLATE" ;
const char* G4DAEPhoton::SHAPE = "4,4" ;
const char* G4DAEPhoton::KEY   = "PHO" ;


void G4DAEPhoton::Collect( G4DAEArrayHolder* photons, const G4Track* aPhoton )
{
   G4ParticleDefinition* pd = aPhoton->GetDefinition();
   assert( pd->GetParticleName() == "opticalphoton" );

   G4String pname="-";
   const G4VProcess* process = aPhoton->GetCreatorProcess();
   if(process) pname = process->GetProcessName();
   assert( pname == "Cerenkov" || pname == "Scintillation" );

   G4ThreeVector pos = aPhoton->GetPosition()/mm ;
   G4ThreeVector dir = aPhoton->GetMomentumDirection() ;
   G4ThreeVector pol = aPhoton->GetPolarization() ;

   const float time = aPhoton->GetGlobalTime()/ns ;
   const float wavelength = (h_Planck * c_light / aPhoton->GetKineticEnergy()) / nanometer ;

   Collect( photons, pos, dir, pol, time, wavelength );
}



void G4DAEPhoton::Collect(G4DAEArrayHolder* photons, const G4ThreeVector& pos, const G4ThreeVector& dir, const G4ThreeVector& pol, const float time, const float wavelength, const int pmtid)
{

    float _weight = 1. ;
    float* data = photons->GetNextPointer();

    data[_post_x] =  pos.x() ;
    data[_post_y] =  pos.y() ;
    data[_post_z] =  pos.z() ;
    data[_post_w] =  time ;

    data[_dirw_x] =  dir.x() ;
    data[_dirw_y] =  dir.y() ;
    data[_dirw_z] =  dir.z() ;
    data[_dirw_w] =  wavelength ;

    data[_polw_x] =  pol.x() ;
    data[_polw_y] =  pol.y() ;
    data[_polw_z] =  pol.z() ;
    data[_polw_w] =  _weight ;

    int _photon_id = 0; 
    int _spare     = 0; 
    unsigned int _flags     = 0 ;

    uif_t uifd[4] ; 
    uifd[0].i = _photon_id ;
    uifd[1].i = _spare ;
    uifd[2].u = _flags     ;
    uifd[3].i =  pmtid     ; 

    data[_flag_x] =  uifd[0].f ;
    data[_flag_y] =  uifd[1].f ;
    data[_flag_z] =  uifd[2].f ;
    data[_flag_w] =  uifd[3].f ;

}

void G4DAEPhoton::Get( G4DAEArrayHolder* photons, std::size_t index , G4ThreeVector& pos, G4ThreeVector& dir, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) 
{
    float* data = photons->GetItemPointer( index );

    pos.setX(data[_post_x]);
    pos.setY(data[_post_y]);
    pos.setZ(data[_post_z]);
    _t = data[G4DAEPhoton::_post_w] ;

    dir.setX(data[_dirw_x]);
    dir.setY(data[_dirw_y]);
    dir.setZ(data[_dirw_z]);
    _wavelength = data[_dirw_w] ;

    pol.setX(data[_polw_x]);
    pol.setY(data[_polw_y]);
    pol.setZ(data[_polw_z]);
    //_weight = data[_polw_w];

    uif_t uifd[4] ; 
    uifd[0].f = data[_flag_x];
    uifd[1].f = data[_flag_y];
    uifd[2].f = data[_flag_z];
    uifd[3].f = data[_flag_w]; 

    // TODO: get this back to caller, struct to hold the quad ?
    int _photon_id ; 
    int _spare ; 
    unsigned int _flags ;

    _photon_id = uifd[0].i ;
    _spare     = uifd[1].i ;
    _flags     = uifd[2].u ;
    _pmtid     = uifd[3].i ;

}



void G4DAEPhoton::Dump(G4DAEArrayHolder* photons, bool /*hit*/) 
{
    size_t count = photons->GetCount();
    cout <<  "G4DAEPhoton::Dump [" << count << "]" << endl ;

    size_t index ;

    G4ThreeVector pos ;
    G4ThreeVector dir ;
    G4ThreeVector pol ;
    float _t ;
    float _wavelength ;
    int _pmtid ;

    for( index = 0 ; index < count ; index++ )
    {
        Get( photons, index , pos, dir, pol, _t, _wavelength, _pmtid );
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


