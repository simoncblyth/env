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
#include "G4DAEChroma/G4DAECommon.hh"


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
    m_verbosity(3),
    m_flags(0)
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



////////  Generate the bitfield flags code from flags.json source using: gdc-flags-gen /////////

const char* G4DAEChroma::_FLAG_G4SCINTILLATION_ADD_SECONDARY    = "FLAG_G4SCINTILLATION_ADD_SECONDARY" ;
const char* G4DAEChroma::_FLAG_G4SCINTILLATION_KILL_SECONDARY   = "FLAG_G4SCINTILLATION_KILL_SECONDARY" ; 
const char* G4DAEChroma::_FLAG_G4SCINTILLATION_COLLECT_STEP     = "FLAG_G4SCINTILLATION_COLLECT_STEP" ; 
const char* G4DAEChroma::_FLAG_G4SCINTILLATION_COLLECT_PHOTON   = "FLAG_G4SCINTILLATION_COLLECT_PHOTON" ; 
const char* G4DAEChroma::_FLAG_G4SCINTILLATION_COLLECT_PROP     = "FLAG_G4SCINTILLATION_COLLECT_PROP" ; 

const char* G4DAEChroma::_FLAG_G4CERENKOV_ADD_SECONDARY         = "FLAG_G4CERENKOV_ADD_SECONDARY" ; 
const char* G4DAEChroma::_FLAG_G4CERENKOV_KILL_SECONDARY        = "FLAG_G4CERENKOV_KILL_SECONDARY" ; 
const char* G4DAEChroma::_FLAG_G4CERENKOV_COLLECT_STEP          = "FLAG_G4CERENKOV_COLLECT_STEP" ; 
const char* G4DAEChroma::_FLAG_G4CERENKOV_COLLECT_PHOTON        = "FLAG_G4CERENKOV_COLLECT_PHOTON" ; 
const char* G4DAEChroma::_FLAG_G4CERENKOV_APPLY_WATER_QE        = "FLAG_G4CERENKOV_APPLY_WATER_QE" ; 

std::string G4DAEChroma::Flags()
{
    std::vector<std::string> elem ; 

    if(HasFlag(FLAG_G4SCINTILLATION_ADD_SECONDARY))      elem.push_back(std::string(_FLAG_G4SCINTILLATION_ADD_SECONDARY)) ;
    if(HasFlag(FLAG_G4SCINTILLATION_KILL_SECONDARY))     elem.push_back(std::string(_FLAG_G4SCINTILLATION_KILL_SECONDARY)) ;
    if(HasFlag(FLAG_G4SCINTILLATION_COLLECT_STEP))       elem.push_back(std::string(_FLAG_G4SCINTILLATION_COLLECT_STEP)) ;
    if(HasFlag(FLAG_G4SCINTILLATION_COLLECT_PHOTON))     elem.push_back(std::string(_FLAG_G4SCINTILLATION_COLLECT_PHOTON)) ;
    if(HasFlag(FLAG_G4SCINTILLATION_COLLECT_PROP))       elem.push_back(std::string(_FLAG_G4SCINTILLATION_COLLECT_PROP)) ;

    if(HasFlag(FLAG_G4CERENKOV_ADD_SECONDARY))           elem.push_back(std::string(_FLAG_G4CERENKOV_ADD_SECONDARY)) ;
    if(HasFlag(FLAG_G4CERENKOV_KILL_SECONDARY))          elem.push_back(std::string(_FLAG_G4CERENKOV_KILL_SECONDARY)) ;
    if(HasFlag(FLAG_G4CERENKOV_COLLECT_STEP))            elem.push_back(std::string(_FLAG_G4CERENKOV_COLLECT_STEP)) ;
    if(HasFlag(FLAG_G4CERENKOV_COLLECT_PHOTON))          elem.push_back(std::string(_FLAG_G4CERENKOV_COLLECT_PHOTON)) ;

    if(HasFlag(FLAG_G4CERENKOV_APPLY_WATER_QE))          elem.push_back(std::string(_FLAG_G4CERENKOV_APPLY_WATER_QE)) ;

    return join(elem, '\n') ; 
}


int G4DAEChroma::MatchFlag(const char* flag )
{

    int ret = FLAG_ZERO ; 

    if(strcmp(flag, _FLAG_G4SCINTILLATION_ADD_SECONDARY ) == 0)  ret = FLAG_G4SCINTILLATION_ADD_SECONDARY ;
    if(strcmp(flag, _FLAG_G4SCINTILLATION_KILL_SECONDARY ) == 0) ret = FLAG_G4SCINTILLATION_KILL_SECONDARY ;
    if(strcmp(flag, _FLAG_G4SCINTILLATION_COLLECT_STEP ) == 0)   ret = FLAG_G4SCINTILLATION_COLLECT_STEP ;
    if(strcmp(flag, _FLAG_G4SCINTILLATION_COLLECT_PHOTON ) == 0) ret = FLAG_G4SCINTILLATION_COLLECT_PHOTON ;
    if(strcmp(flag, _FLAG_G4SCINTILLATION_COLLECT_PROP ) == 0)   ret = FLAG_G4SCINTILLATION_COLLECT_PROP ;

    if(strcmp(flag, _FLAG_G4CERENKOV_ADD_SECONDARY ) == 0)       ret = FLAG_G4CERENKOV_ADD_SECONDARY ;
    if(strcmp(flag, _FLAG_G4CERENKOV_KILL_SECONDARY ) == 0)      ret = FLAG_G4CERENKOV_KILL_SECONDARY ;
    if(strcmp(flag, _FLAG_G4CERENKOV_COLLECT_STEP ) == 0)        ret = FLAG_G4CERENKOV_COLLECT_STEP ;
    if(strcmp(flag, _FLAG_G4CERENKOV_COLLECT_PHOTON ) == 0)      ret = FLAG_G4CERENKOV_COLLECT_PHOTON ;

    if(strcmp(flag, _FLAG_G4CERENKOV_APPLY_WATER_QE ) == 0)      ret = FLAG_G4CERENKOV_APPLY_WATER_QE ;

    //cout << "G4DAEChroma::MatchFlag " << flag << " " << ret << endl ;

    return ret ; 
}
/////////////// end of generatable code //////////////////




int G4DAEChroma::ParseFlags(std::string sflags, char delim)
{

    //cout << "G4DAEChroma::ParseFlags " << sflags << endl ; 

    typedef std::vector<std::string> Vec_t ;
    Vec_t elems ; 
    split(elems, sflags.c_str(), delim);

    int flags = 0 ; 
    for(Vec_t::iterator it=elems.begin() ; it!=elems.end() ; it++)
    {
        std::string elem = *it;
        int eflag = MatchFlag( elem.c_str() );
        cout << "G4DAEChroma::ParseFlags  elem " << elem << " eflag " << eflag << endl ; 
        flags |= eflag ; 
    }
    return flags ; 
}



void G4DAEChroma::SetFlags(std::string flags)
{
     int _flags = ParseFlags(flags);
     SetFlags(_flags);
}
void G4DAEChroma::SetFlags(int flags)
{
    m_flags = flags ; 
}
int G4DAEChroma::GetFlags()
{
    return m_flags ; 
}
void G4DAEChroma::AddFlags(int flags)
{
    m_flags |= flags ; 
}
void G4DAEChroma::AddFlags(std::string flags)
{
    int _flags = ParseFlags(flags);
    AddFlags(_flags); 
}




bool G4DAEChroma::HasFlag(int flag)
{
    return m_flags & flag ; 
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
    cout << "flags:\n"     << Flags() << endl ;
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



void G4DAEChroma::SetHCofThisEvent(G4HCofThisEvent* hce){
   m_hce = hce ; 
}
G4HCofThisEvent* G4DAEChroma::GetHCofThisEvent(){
   return m_hce ; 
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
G4DAECollector* G4DAEChroma::GetCollector(){
   if(!m_sensdet) return NULL ;
   return m_sensdet->GetCollector(); 
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
G4DAEPmtHitList* G4DAEChroma::GetPmtHitList()
{
   return m_transport->GetPmtHitList();
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




std::size_t G4DAEChroma::ProcessCerenkovPhotons(G4int batch_id)
{
    std::size_t nhits = m_transport->ProcessCerenkovPhotons(batch_id); 
    return nhits ; 
}

std::size_t G4DAEChroma::ProcessScintillationPhotons(G4int batch_id)
{
    std::size_t nhits = m_transport->ProcessScintillationPhotons(batch_id); 
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
  else
  {
      cout << "G4DAEChroma::Propagate : zero hits   " << endl ; 

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


