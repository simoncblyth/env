#include "G4DAEChroma/G4DAEChroma.hh"

#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAEDatabase.hh"

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
    m_database(0)
{ 
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







void G4DAEChroma::CollectPhoton(const G4Track* track)
{
   m_transport->CollectPhoton(track);
}



std::size_t G4DAEChroma::Propagate(G4int batch_id)
{
#ifdef VERBOSE
  cout << "G4DAEChroma::Propagate START batch_id " << batch_id << endl ; 
#endif
  std::size_t nhits = m_transport->Propagate(batch_id); 
#ifdef VERBOSE
  cout << "G4DAEChroma::Propagate CollectHits batch_id " << batch_id << endl ; 
#endif
  if(nhits > 0)
  { 
      m_sensdet->CollectHits( m_transport->GetHits(), m_cache );
  } 
#ifdef VERBOSE
  cout << "G4DAEChroma::Propagate DONE batch_id " << batch_id << " nhits " << nhits << endl ; 
#endif
  return nhits ; 
}





