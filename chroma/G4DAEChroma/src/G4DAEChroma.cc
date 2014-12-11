#include "G4DAEChroma/G4DAEChroma.hh"

#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAEDatabase.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEHitList.hh"
#include "G4DAEChroma/G4DAECollector.hh"
#include "G4DAEChroma/G4DAEPhotons.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"

#include "G4ThreeVector.hh"
#include "G4AffineTransform.hh"

#include <iostream>


using namespace std ; 

G4DAEChroma* G4DAEChroma::fG4DAEChroma = 0;


G4DAEChroma* G4DAEChroma::GetG4DAEChroma()
{
  if(!fG4DAEChroma)
  {
     fG4DAEChroma = new G4DAEChroma;
  }
  return fG4DAEChroma;
}

G4DAEChroma* G4DAEChroma::GetG4DAEChromaIfExists()
{ 
  return fG4DAEChroma ;
}


G4DAEChroma::G4DAEChroma() :
    m_transport(0),
    m_sensdet(0),
    m_geometry(0),
    m_cache(0),
    m_database(0),
    m_verbosity(3)
{ 
}

void G4DAEChroma::Print(const char* msg)
{
    cout << msg << endl ; 
    cout << "transport " << m_transport << endl ; 
    cout << "sensdet   " << m_sensdet   << endl ; 
    cout << "geometry  " << m_geometry  << endl ; 
    cout << "cache     " << m_cache     << endl ; 
    cout << "database  " << m_database  << endl ; 
    cout << "verbosity " << m_verbosity << endl ; 
}


void G4DAEChroma::BeginOfRun( const G4Run* run )
{
    cout << "G4DAEChroma::BeginOfRun [" << this << "] " << run << endl ;
}
void G4DAEChroma::EndOfRun(   const G4Run* run )
{
    cout << "G4DAEChroma::EndOfRun [" << this << "] " << run << endl ;
}

void G4DAEChroma::Configure(const char* transport, const char* sensdet, const char* geometry, const char* database)
{
    cout << "G4DAEChroma::Configure [" << this << "]" << endl ;
    G4DAETransport* tra = new G4DAETransport(transport);
    G4DAEGeometry*  geo = G4DAEGeometry::MakeGeometry(geometry);
    G4DAEDatabase*  db = new G4DAEDatabase(database);

    const char* target = sensdet ; 
    string trojan = "trojan_" ;
    trojan += sensdet ;
    G4DAESensDet*   tsd = G4DAESensDet::MakeSensDet(trojan.c_str(), target );


    this->SetTransport( tra );
    this->SetGeometry( geo );  
    this->SetSensDet( tsd );  
    this->SetDatabase( db );  
}

void G4DAEChroma::Note(const char* msg)
{
    cout << "G4DAEChroma::Note [" <<  this << "] " << msg << endl ;
}


G4DAEChroma::~G4DAEChroma()
{
    delete m_transport ;
    delete m_sensdet ;
    delete m_geometry ;
    delete m_cache ;
    delete m_database ;
}

void G4DAEChroma::SetTransport(G4DAETransport* tra){
   m_transport = tra ; 
}
G4DAETransport* G4DAEChroma::GetTransport(){
   return m_transport ;
}


void G4DAEChroma::SetSensDet(G4DAESensDet* sd){
   m_sensdet = sd ; 
}
G4DAESensDet* G4DAEChroma::GetSensDet(){
   return m_sensdet ;
}

void G4DAEChroma::SetGeometry(G4DAEGeometry* geo){
   m_geometry = geo ; 
}
G4DAEGeometry* G4DAEChroma::GetGeometry(){
   return m_geometry ;
}

void G4DAEChroma::SetTransformCache(G4DAETransformCache* cache){
   m_cache = cache ; 
}
G4DAETransformCache* G4DAEChroma::GetTransformCache(){
   return m_cache ;
}

void G4DAEChroma::SetDatabase(G4DAEDatabase* db){
   m_database = db ; 
}
G4DAEDatabase* G4DAEChroma::GetDatabase(){
   return m_database ;
}


void G4DAEChroma::SetVerbosity(int verbosity){
   m_verbosity = verbosity ;
}
int G4DAEChroma::GetVerbosity(){
   return m_verbosity ;
}




#ifdef DEBUG_HITLIST
G4DAEHitList* G4DAEChroma::GetHitList()
{
   G4DAESensDet* sd = GetSensDet(); 
   return sd->GetCollector()->GetHits(); 
   // G4DAEHitList only adds local coords compared to G4DAEPhotons
}
#endif




void G4DAEChroma::SetPhotons(G4DAEPhotons* photons)
{
   m_transport->SetPhotons(photons);
}
void G4DAEChroma::SetHits(G4DAEPhotons* hits)
{
   m_transport->SetHits(hits);
}


void G4DAEChroma::SavePhotons(const char* evtkey )
{
   G4DAEPhotons* photons = m_transport->GetPhotons();
   photons->Print("G4DAEChroma::SavePhotons");
   G4DAEPhotons::Save( photons, evtkey ); 
}

void G4DAEChroma::LoadPhotons(const char* evtkey )
{
   G4DAEPhotons* photons = G4DAEPhotons::Load(evtkey ); 
   photons->Print("G4DAEChroma::SavePhotons");
   m_transport->SetPhotons(photons);  // leaking prior photons
}




G4DAEPhotons* G4DAEChroma::GetPhotons()
{
   return m_transport->GetPhotons();
}
G4DAEPhotons* G4DAEChroma::GetHits()
{
   return m_transport->GetHits();
}

void G4DAEChroma::CollectPhoton(const G4Track* track)
{
   m_transport->CollectPhoton(track);
}

void G4DAEChroma::ClearAll()
{
    m_transport->ClearAll();
}


G4DAEPhotons* G4DAEChroma::Propagate(G4DAEPhotons* photons)
{
   m_transport->SetPhotons(photons);
   std::size_t nhits = this->Propagate(1); // >0 for real propagation, otherwise fakes
   printf("G4DAEChroma::Propagate returned %zu hits \n", nhits); 
   return m_transport->GetHits();
}




std::size_t G4DAEChroma::Propagate(G4int batch_id)
{
  if(m_verbosity > 1)
      cout << "G4DAEChroma::Propagate START batch_id " << batch_id << endl ; 

  std::size_t nhits = m_transport->Propagate(batch_id); 

  if(m_verbosity > 1) 
      cout << "G4DAEChroma::Propagate CollectHits batch_id " << batch_id << endl ; 

  if(nhits > 0)
  { 
      G4DAEPhotons* hits = m_transport->GetHits() ;
      if(m_verbosity > 1){
          hits->Print("G4DAEChroma::Propagate returned hits"); 
          hits->Details(1); 
      }
      m_sensdet->CollectHits( hits, m_cache );
  } 

  if(m_verbosity > 1)
      cout << "G4DAEChroma::Propagate DONE batch_id " << batch_id << " nhits " << nhits << endl ; 

  return nhits ; 
}





G4DAEPhotons* G4DAEChroma::GenerateMockPhotons()
{
    G4DAETransformCache* cache = GetTransformCache();
    if(!cache) return NULL ; 
    size_t size = cache->GetSize();


    G4DAEPhotons* photons = (G4DAEPhotons*)new G4DAEPhotonList(size);

    G4ThreeVector lpos(0,0,1500) ;  
    G4ThreeVector ldir(0,0,-1) ;
    G4ThreeVector lpol(0,0,1) ; 
    const float time = 1. ;
    const float wavelength = 550. ;

    size_t count = 0 ;
    for( size_t index = 0 ; index < cache->GetSize() ; ++index ) // cache contains affine transforms for all PMTs
    {
        if( index % 1 == 0 && count < size )
        {
            int pmtid = cache->GetKey(index);

            G4AffineTransform* pg2l = cache->GetSensorTransform(pmtid);
            assert(pg2l);

            G4AffineTransform g2l(*pg2l);
            G4AffineTransform l2g(g2l.Inverse());
           
            G4ThreeVector gpos(l2g.TransformPoint(lpos));
            G4ThreeVector gdir(l2g.TransformAxis(ldir));
            G4ThreeVector gpol(l2g.TransformAxis(lpol));

            photons->AddPhoton( gpos, gdir, gpol, time, wavelength, pmtid );
            count++ ;
        }
    } 
    return photons ; 
}


