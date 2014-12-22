#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAESocketBase.hh"

#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAECerenkovStepList.hh"
#include "G4DAEChroma/G4DAEScintillationStepList.hh"
#include "G4DAEChroma/G4DAEFotonList.hh"
#include "G4DAEChroma/G4DAEXotonList.hh"

#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEMap.hh"
#include "G4DAEChroma/G4DAECommon.hh"

#include "G4Track.hh"
#include "G4VProcess.hh"

#include <string>
#include <iostream>
#include <stdio.h>


#define G4DAETRANSPORT_VERBOSE

using namespace std ; 

G4DAETransport::G4DAETransport(const char* envvar) :
    m_socket(NULL),
    m_photons(NULL),
    m_hits(NULL),
    m_cerenkov(NULL),
    m_scintillation(NULL),
    m_fotons(NULL),
    m_xotons(NULL),
    m_handshake(NULL),
    m_verbosity(3)
{ 
#ifdef WITH_CHROMA_ZMQ
   m_socket = new G4DAESocketBase(envvar) ;

   // the arrays grow as needed

   m_photons = new G4DAEPhotonList(10000) ;   

   m_cerenkov = new G4DAECerenkovStepList(10000);

   m_scintillation = new G4DAEScintillationStepList(10000);

   m_fotons = new G4DAEFotonList(10000);

   m_xotons = new G4DAEXotonList(10000);

#endif
}

G4DAETransport::~G4DAETransport()
{
#ifdef WITH_CHROMA_ZMQ
   delete m_photons ; 
   delete m_hits  ; 
   delete m_socket ; 
   delete m_cerenkov ; 
   delete m_scintillation ; 
   delete m_fotons ; 
   delete m_xotons ; 
   delete m_handshake ; 
#endif
}


int G4DAETransport::GetVerbosity()
{
    return m_verbosity ;
}

void G4DAETransport::SetVerbosity(int verbosity)
{
     m_verbosity = verbosity ;
}



G4DAEMetadata* G4DAETransport::GetHandshake()
{ 
    return m_handshake ; 
}

void G4DAETransport::Handshake(G4DAEMetadata* request)
{
    if(!request) request = new G4DAEMetadata("{}"); 

    if( m_verbosity > 0 )
    request->Print("G4DAETransport::Handshake waiting for handshake response:");

    m_handshake = reinterpret_cast<G4DAEMetadata*>(m_socket->SendReceiveObject(request));

    if(!m_handshake)
    {
        request->Print("G4DAETransport::Handshake FAILED with request:");
        return ;  
    }

    //m_handshake->PrintMap("G4DAETransport::Handshake PrintMap");
}




G4DAEPhotonList* G4DAETransport::GetPhotons(){ 
    return m_photons ; 
}
G4DAEPhotonList* G4DAETransport::GetHits(){ 
    return m_hits ; 
}
G4DAECerenkovStepList* G4DAETransport::GetCerenkovStepList(){ 
    return m_cerenkov ; 
}
G4DAEScintillationStepList* G4DAETransport::GetScintillationStepList(){ 
    return m_scintillation ; 
}
G4DAEFotonList* G4DAETransport::GetFotonList(){ 
    return m_fotons ; 
}
G4DAEXotonList* G4DAETransport::GetXotonList(){ 
    return m_xotons ; 
}






void G4DAETransport::SetPhotons(G4DAEPhotonList* photons)
{
   delete m_photons ; 
   m_photons = photons ; 
}
void G4DAETransport::SetHits(G4DAEPhotonList* hits)
{
   delete m_hits ; 
   m_hits = hits ; 
}
void G4DAETransport::SetCerenkovStepList(G4DAECerenkovStepList* cerenkov)
{
   delete m_cerenkov ; 
   m_cerenkov = cerenkov ; 
}
void G4DAETransport::SetScintillationStepList(G4DAEScintillationStepList* scintillation)
{
   delete m_scintillation ; 
   m_scintillation = scintillation ; 
}
void G4DAETransport::SetFotonList(G4DAEFotonList* fotons)
{
   delete m_fotons ; 
   m_fotons = fotons ; 
}









void G4DAETransport::ClearAll()
{
#ifdef WITH_CHROMA_ZMQ
    if(m_photons)
    {
        m_photons->ClearAll();   
    }
    if(m_hits)
    {
        m_hits->ClearAll();   
    }
    if(m_cerenkov)
    {
        m_cerenkov->ClearAll();   
    }
    if(m_scintillation)
    {
        m_scintillation->ClearAll();   
    }
    if(m_fotons)
    {
        m_fotons->ClearAll();   
    }
    if(m_xotons)
    {
        m_xotons->ClearAll();   
    }
#endif
}


void G4DAETransport::CollectPhoton(const G4Track* aPhoton )
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

   CollectPhoton( pos, dir, pol, time, wavelength );
}





std::size_t G4DAETransport::ProcessCerenkovSteps(int batch_id)
{
    return Process(batch_id, m_cerenkov );
}

std::size_t G4DAETransport::ProcessScintillationSteps(int batch_id)
{
    return Process(batch_id, m_scintillation );
}

std::size_t G4DAETransport::Propagate(int batch_id)
{
    return Process(batch_id, m_photons );
}


std::size_t G4DAETransport::Process(int batch_id, G4DAEArrayHolder* request)
{
   size_t size = request ? request->GetCount() : 0 ;
   if(size == 0){
       request->Print("G4DAETransport::Process EMPTY request");
       return 0 ;
   }

   if(m_verbosity > 1)
   {
       request->Print("G4DAETransport::Process");
   } 

   if( m_verbosity > 0 ) cout << "G4DAETransport::Process : SendReceiveObject batch_id " << batch_id <<  endl ;   

   m_hits = reinterpret_cast<G4DAEPhotonList*>(m_socket->SendReceiveObject(request));

   std::size_t nhits = m_hits ? m_hits->GetCount() : 0 ;
   return nhits ; 

}






void G4DAETransport::CollectPhoton(const G4ThreeVector& pos, const G4ThreeVector& dir, const G4ThreeVector& pol, const float time, const float wavelength, const int pmtid)
{

    float _weight = 1. ;
    float* data = m_photons->GetNextPointer();

    data[G4DAEPhoton::_post_x] =  pos.x() ;
    data[G4DAEPhoton::_post_y] =  pos.y() ;
    data[G4DAEPhoton::_post_z] =  pos.z() ;
    data[G4DAEPhoton::_post_w] =  time ;

    data[G4DAEPhoton::_dirw_x] =  dir.x() ;
    data[G4DAEPhoton::_dirw_y] =  dir.y() ;
    data[G4DAEPhoton::_dirw_z] =  dir.z() ;
    data[G4DAEPhoton::_dirw_w] =  wavelength ;

    data[G4DAEPhoton::_polw_x] =  pol.x() ;
    data[G4DAEPhoton::_polw_y] =  pol.y() ;
    data[G4DAEPhoton::_polw_z] =  pol.z() ;
    data[G4DAEPhoton::_polw_w] =  _weight ;

    int _photon_id = 0; 
    int _spare     = 0; 
    unsigned int _flags     = 0 ;

    uif_t uifd[4] ; 
    uifd[0].i = _photon_id ;
    uifd[1].i = _spare ;
    uifd[2].u = _flags     ;
    uifd[3].i =  pmtid     ; 

    data[G4DAEPhoton::_flag_x] =  uifd[0].f ;
    data[G4DAEPhoton::_flag_y] =  uifd[1].f ;
    data[G4DAEPhoton::_flag_z] =  uifd[2].f ;
    data[G4DAEPhoton::_flag_w] =  uifd[3].f ;

}

void G4DAETransport::GetPhoton( std::size_t index , G4ThreeVector& pos, G4ThreeVector& dir, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const
{
    float* data = m_photons->GetItemPointer( index );

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



void G4DAETransport::DumpPhotons(bool /*hit*/) const 
{
    size_t count = m_photons->GetCount();
    cout <<  "G4DAETransport::DumpPhotons [" << count << "]" << endl ;

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


