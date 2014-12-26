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
#include "G4DAEChroma/G4DAECerenkovStepList.hh"
#include "G4DAEChroma/G4DAEScintillationStepList.hh"
#include "G4DAEChroma/G4DAEMaterialMap.hh"


#include "G4ThreeVector.hh"
#include "G4AffineTransform.hh"

#include <iostream>
#include <assert.h>


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
    m_metadata(0),
    m_materialmap(0),
    m_g2c(0),
    m_g4cerenkov(false),
    m_g4scintillation(false),
    m_verbosity(3)
{ 
}

G4DAEChroma::~G4DAEChroma()
{
    delete m_transport ;
    delete m_sensdet ;
    delete m_geometry ;
    delete m_cache ;
    delete m_database ;
    delete m_metadata ;
    delete m_materialmap ;
    delete m_g2c;
}

void G4DAEChroma::SetG4Cerenkov(bool do_)
{
   m_g4cerenkov = do_ ; 
}
void G4DAEChroma::SetG4Scintillation(bool do_)
{
   m_g4scintillation = do_ ; 
}
bool G4DAEChroma::IsG4Cerenkov()
{
   return m_g4cerenkov ; 
}
bool G4DAEChroma::IsG4Scintillation()
{
   return m_g4scintillation ; 
}





void G4DAEChroma::Print(const char* msg)
{
    cout << msg << endl ; 
    cout << "transport   " << m_transport << endl ; 
    cout << "sensdet     " << m_sensdet   << endl ; 
    cout << "geometry    " << m_geometry  << endl ; 
    cout << "cache       " << m_cache     << endl ; 
    cout << "database    " << m_database  << endl ; 
    cout << "metadata    " << m_metadata  << endl ; 
    cout << "materialmap " << m_materialmap  << endl ; 
    cout << "g2c         " << m_g2c  << endl ; 
    cout << "verbosity   " << m_verbosity  << endl ; 
    cout << "g4cerenkov        " << m_g4cerenkov << endl ; 
    cout << "g4scintillation   " << m_g4scintillation  << endl ; 
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
    assert(0); // not in use : TODO:get rid of this

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

void G4DAEChroma::Handshake(G4DAEMetadata* request)
{
    if(!m_transport) return;
    m_transport->Handshake(request);
}

G4DAEMetadata* G4DAEChroma::GetHandshake()
{
    if(!m_transport) return NULL;
    return m_transport->GetHandshake();
}





void G4DAEChroma::SetMaterialMap(G4DAEMaterialMap* map){
   m_materialmap = map ; 
}
G4DAEMaterialMap* G4DAEChroma::GetMaterialMap(){
   return m_materialmap ;
}
void G4DAEChroma::SetMaterialLookup(int* g2c){
   m_g2c = g2c ; 
}
int* G4DAEChroma::GetMaterialLookup(){
   return m_g2c ;
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

void G4DAEChroma::SetMetadata(G4DAEMetadata* meta){
   m_metadata = meta ; 
}
G4DAEMetadata* G4DAEChroma::GetMetadata(){
   return m_metadata ;
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




void G4DAEChroma::SetPhotons(G4DAEPhotonList* photons)
{
   m_transport->SetPhotons(photons);
}
void G4DAEChroma::SetHits(G4DAEPhotonList* hits)
{
   m_transport->SetHits(hits);
}


void G4DAEChroma::SavePhotons(const char* evtkey )
{
   G4DAEPhotonList* photons = m_transport->GetPhotons();
   photons->Print("G4DAEChroma::SavePhotons");
   photons->Save( evtkey ); 
}

void G4DAEChroma::LoadPhotons(const char* evtkey )
{
   G4DAEList<G4DAEPhoton>* pl = G4DAEPhotonList::Load(evtkey ); 
   G4DAEPhotonList* photons = reinterpret_cast<G4DAEPhotonList*>(pl);
   photons->Print("G4DAEChroma::SavePhotons");
   m_transport->SetPhotons(photons);  // leaking prior photons
}





G4DAECerenkovStepList* G4DAEChroma::GetCerenkovStepList()
{
   return m_transport->GetCerenkovStepList();
}
G4DAEScintillationStepList* G4DAEChroma::GetScintillationStepList()
{
   return m_transport->GetScintillationStepList();
}
G4DAEScintillationPhotonList* G4DAEChroma::GetScintillationPhotonList()
{
   return m_transport->GetScintillationPhotonList();
}
G4DAECerenkovPhotonList* G4DAEChroma::GetCerenkovPhotonList()
{
   return m_transport->GetCerenkovPhotonList();
}







G4DAEPhotonList* G4DAEChroma::GetPhotons()
{
    return m_transport->GetPhotons();
}
G4DAEPhotonList* G4DAEChroma::GetHits()
{
    return m_transport->GetHits();
}

void G4DAEChroma::CollectPhoton(const G4Track* track)
{
    G4DAEPhoton::Collect( m_transport->GetPhotons(),  track );
}

void G4DAEChroma::ClearAll()
{
    m_transport->ClearAll();
}


G4DAEPhotonList* G4DAEChroma::Propagate(G4DAEPhotonList* photons)
{
    m_transport->SetPhotons(photons);
    std::size_t nhits = this->Propagate(1); // >0 for real propagation, otherwise fakes
    printf("G4DAEChroma::Propagate returned %zu hits \n", nhits); 
    return m_transport->GetHits();
}



std::size_t G4DAEChroma::ProcessCerenkovSteps(G4int batch_id)
{
    std::size_t nhits = m_transport->ProcessCerenkovSteps(batch_id); 
    return nhits ; 
}

std::size_t G4DAEChroma::ProcessScintillationSteps(G4int batch_id)
{
    std::size_t nhits = m_transport->ProcessScintillationSteps(batch_id); 
    return nhits ; 
}


std::size_t G4DAEChroma::Propagate(G4int batch_id)
{
   // remember that may do multiple propagations for 
   // for a single event, so this is not the place 
   // for end of event activities


  if(m_verbosity > 1)
      cout << "G4DAEChroma::Propagate START batch_id " << batch_id << endl ; 

 
  G4DAEPhotonList* photons = m_transport->GetPhotons();
  G4DAEMetadata* phometa = new G4DAEMetadata("{}") ; 
  if(m_database)
  {  
      int cid = 2 ; // TODO: make this a parameter somehow
      Map_t ctrl = m_database->GetOne("select * from ctrl where id=? ;", cid ) ;
      phometa->AddMap("ctrl", ctrl);
  }
  phometa->Print("#phometa");
  photons->AddLink(phometa);


  if(m_verbosity > 1)
      photons->Print("G4DAEChroma::Propagate photons"); 


  std::size_t nhits = m_transport->Propagate(batch_id); 


  if(m_verbosity > 1) 
      cout << "G4DAEChroma::Propagate CollectHits batch_id " << batch_id << endl ; 

  if(nhits > 0)
  { 
      G4DAEPhotonList* hits = m_transport->GetHits() ;
      G4DAEMetadata* hitmeta = hits->GetLink();

      if(m_verbosity > 1)
      {
          hits->Print("G4DAEChroma::Propagate returned hits"); 
      }
      m_sensdet->CollectHits( hits, m_cache );


      if(m_metadata && hitmeta)
      {
          m_metadata->AddLink(hitmeta);     // **Add** not **Set** : tacks on to last link of chain
          // TODO: clear metadata at end of event   
      } 
      else
      {
           cout << "G4DAEChroma::Propagate missing m_metadata " << m_metadata << " or hitmeta " << hitmeta << endl ;
      } 

  } 

  if(m_verbosity > 1)
      cout << "G4DAEChroma::Propagate DONE batch_id " << batch_id << " nhits " << nhits << endl ; 

  return nhits ; 
}





G4DAEPhotonList* G4DAEChroma::GenerateMockPhotons()
{
    G4DAETransformCache* cache = GetTransformCache();
    if(!cache) return NULL ; 
    size_t size = cache->GetSize();


    G4DAEPhotonList* photons = new G4DAEPhotonList(size);

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

            G4DAEPhoton::Collect( photons, gpos, gdir, gpol, time, wavelength, pmtid );
            count++ ;
        }
    } 
    return photons ; 
}


