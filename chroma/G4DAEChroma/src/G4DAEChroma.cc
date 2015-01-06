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
    G4DAEManager("G4DAECHROMA_CONFIG_PATH"),
    m_transport(0),
    m_sensdet(0),
    m_trojan_sensdet(0),
    m_active_sensdet(0),
    m_geometry(0),
    m_cache(0),
    m_database(0),
    m_metadata(0),
    m_materialmap(0),
    m_g2c(0),
    m_verbosity(3),
    m_cid(0)
{ 
}

G4DAEChroma::~G4DAEChroma()
{
    delete m_transport ;
    delete m_sensdet ;
    delete m_trojan_sensdet ;
    //delete m_active_sensdet ;  just points to one of the other two
    delete m_geometry ;
    delete m_cache ;
    delete m_database ;
    delete m_metadata ;
    delete m_materialmap ;
    delete m_g2c;
}




void G4DAEChroma::Print(const char* msg)
{
    cout << msg << endl ; 
    cout << "transport       " << m_transport << endl ; 
    cout << "sensdet         " << m_sensdet   << endl ; 
    cout << "trojan_sensdet  " << m_trojan_sensdet   << endl ; 
    cout << "active_sensdet  " << m_trojan_sensdet   << endl ; 
    cout << "geometry        " << m_geometry  << endl ; 
    cout << "cache           " << m_cache     << endl ; 
    cout << "database        " << m_database  << endl ; 
    cout << "metadata        " << m_metadata  << endl ; 
    cout << "materialmap     " << m_materialmap  << endl ; 
    cout << "g2c             " << m_g2c  << endl ; 
    cout << "verbosity       " << m_verbosity  << endl ; 
    cout << "cid             " << m_cid << endl ; 
    cout << "flags:\n"     << Flags() << endl ;

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




void G4DAEChroma::SetHCofThisEvent(G4HCofThisEvent* hce){
   m_hce = hce ; 
}
G4HCofThisEvent* G4DAEChroma::GetHCofThisEvent(){
   return m_hce ; 
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


void G4DAEChroma::SetTrojanSensDet(G4DAESensDet* tsd){
   m_trojan_sensdet = tsd ; 
}
G4DAESensDet* G4DAEChroma::GetTrojanSensDet(){
   return m_trojan_sensdet ;
}
G4DAECollector* G4DAEChroma::GetTrojanCollector(){
   if(!m_trojan_sensdet) return NULL ;
   return m_trojan_sensdet->GetCollector(); 
}


G4DAESensDet* G4DAEChroma::GetActiveSensDet(){
   return m_active_sensdet ;
}
void G4DAEChroma::SetActiveSensDet(G4DAESensDet* asd){
   m_active_sensdet = asd ; 
}
G4DAECollector* G4DAEChroma::GetActiveCollector(){
   if(!m_active_sensdet) return NULL ;
   return m_active_sensdet->GetCollector(); 
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



void G4DAEChroma::SetEvent(int evt){
   m_event = evt ;
}
int G4DAEChroma::GetEvent(){
   return m_event ;
}

void G4DAEChroma::SetControlId(int cid){
   m_cid = cid ;
}
int G4DAEChroma::GetControlId(){
   return m_cid ;
}

void G4DAEChroma::SetVerbosity(int verbosity){
   m_verbosity = verbosity ;
}
int G4DAEChroma::GetVerbosity(){
   return m_verbosity ;
}





void G4DAEChroma::SetPhotonList(G4DAEPhotonList* photons)
{
   m_transport->SetPhotonList(photons);
}
void G4DAEChroma::SetHits(G4DAEPhotonList* hits)
{
   m_transport->SetHits(hits);
}


void G4DAEChroma::SavePhotons(const char* evtkey )
{
   G4DAEPhotonList* photons = m_transport->GetPhotonList();
   photons->Print("G4DAEChroma::SavePhotons");
   photons->Save( evtkey ); 
}

void G4DAEChroma::LoadPhotons(const char* evtkey )
{
   G4DAEList<G4DAEPhoton>* pl = G4DAEPhotonList::Load(evtkey ); 
   G4DAEPhotonList* photons = reinterpret_cast<G4DAEPhotonList*>(pl);
   photons->Print("G4DAEChroma::LoadPhotons");
   m_transport->SetPhotonList(photons);  // leaking prior photons
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
G4DAEPhotonList* G4DAEChroma::GetPhotonList()
{
    return m_transport->GetPhotonList();
}


G4DAEPhotonList* G4DAEChroma::GetHits()
{
    return m_transport->GetHits();
}

void G4DAEChroma::CollectPhoton(const G4Track* track)
{
    G4DAEPhoton::Collect( m_transport->GetPhotonList(),  track );
}

void G4DAEChroma::ClearAll()
{
    m_transport->ClearAll();
}




std::size_t G4DAEChroma::ProcessCerenkovSteps(int evt)
{
    G4DAECerenkovStepList* req = m_transport->GetCerenkovStepList(); 

    req->SetKV("ctrl", "type", "cerenkov" );
    req->SetKV("ctrl", "evt", evt );
    req->SetKV("ctrl", "threads_per_block", 512 );
    req->SetKV("ctrl", "noreturn", 0 );
    req->SetKV("ctrl", "sidesave", 1 );   // remote save of GPU generated photons

    //AttachControlMetadata(req);
    return ProcessSteps(req);
}
std::size_t G4DAEChroma::ProcessScintillationSteps(int evt)
{
    G4DAEScintillationStepList* req = m_transport->GetScintillationStepList(); 

    req->SetKV("ctrl", "type", "scintillation" );
    req->SetKV("ctrl", "evt", evt );
    req->SetKV("ctrl", "threads_per_block", 512 );
    req->SetKV("ctrl", "noreturn", 0 );
    req->SetKV("ctrl", "sidesave", 1 );   // remote save of GPU generated photons

    //AttachControlMetadata(req);
    return ProcessSteps(req);
}


std::size_t G4DAEChroma::ProcessSteps(G4DAEArrayHolder* steps)
{
    G4DAEArrayHolder* response = m_transport->Process(steps);
    G4DAEPhotonList* hits = new G4DAEPhotonList(response);
    SetHits(hits);

    assert(hits);
    hits->Print("response from ProcessSteps ");

    return CollectHits(hits);
}





std::size_t G4DAEChroma::ProcessCerenkovPhotons(int evt)
{
    G4DAECerenkovPhotonList* req = m_transport->GetCerenkovPhotonList(); 
    CopyToRemote(req, evt, "gopcerenkov");
    return 0 ;
}

std::size_t G4DAEChroma::ProcessScintillationPhotons(int evt)
{
    G4DAEScintillationPhotonList* req = m_transport->GetScintillationPhotonList(); 
    CopyToRemote(req, evt, "gopcerenkov");
    return 0 ;
}


void G4DAEChroma::CopyToRemote(G4DAEArrayHolder* req, int evt, const char* type)
{
    printf("G4DAEChroma::CopyToRemote  evt %d type %s \n", evt, type);

    req->SetKV("ctrl", "type", type );
    req->SetKV("ctrl", "evt", evt );
    req->SetKV("ctrl", "onlycopy", 1 );

    G4DAEArrayHolder* response = m_transport->Process(req);
    if( response )  // will be NULL when req is empty 
    {
        assert( response->GetCount() == 0 );
    }
}




std::size_t G4DAEChroma::ProcessCollectedPhotons(int /*evt*/)
{
    G4DAEPhotonList* req = m_transport->GetPhotonList(); 
    //AttachControlMetadata(req);
    return ProcessPhotons(req);
}


std::size_t G4DAEChroma::ProcessPhotons(G4DAEArrayHolder* photons)
{
    G4DAEArrayHolder* response = m_transport->Process(photons);
    G4DAEPhotonList* hits = new G4DAEPhotonList(response);
    SetHits(hits);
    return CollectHits(hits);
}



std::size_t G4DAEChroma::CollectHits(G4DAEPhotonList* hits)
{
    G4DAESensDet* sd = GetActiveSensDet();
    size_t nhits = 0 ; 
    if( sd == NULL )
    {
        printf("G4DAEChroma::CollectHits WARNING there is no active SensDet, cannot collect hits \n"); 
    }
    else
    {
        sd->CollectHits( hits, m_cache );  // hits ->  G4 hit collections
        nhits = hits->GetCount(); 
    } 
    return nhits ; 
}


void G4DAEChroma::AttachControlMetadata(G4DAEArrayHolder* request)
{
    G4DAEMetadata* meta = new G4DAEMetadata("{}") ; 
    if(m_database)
    {  
        int cid = GetControlId();
        Map_t ctrl = m_database->GetOne("select * from ctrl where id=? ;", cid ) ;
        meta->AddMap("ctrl", ctrl);
    }
    meta->Print("#request");
    request->AddLink(meta);
}










void G4DAEChroma::BeginOfRun( const G4Run* /*run*/ )
{
    Start("RUN", 1);
}
void G4DAEChroma::EndOfRun(   const G4Run* /*run*/ )
{
    Stop("RUN", 1);
}
void G4DAEChroma::BeginOfEvent(const G4Event* /*event*/)
{
    SetEvent(1);
    Stamp("BeginOfEvent", 1);
    Start("EVENT", 1);
}
void G4DAEChroma::EndOfEvent(const G4Event* /*event*/)
{
    Stop("EVENT", 1);
    Stamp("EndOfEvent", 1);

    int evt = GetEvent();
    GPUProcessing(evt);
    PostProcessing(evt);
}


void G4DAEChroma::PostProcessing(int evt)
{
    
    {
        G4DAESensDet* tsd = GetTrojanSensDet();

        G4DAEPmtHitList* phl = new G4DAEPmtHitList(1000);

        tsd->PopulatePmtHitList(phl);  // from G4 HC into PmtHitList

        phl->Print("G4DAEChroma::PostProcessing from TrojanSD");

        //phl->Save(evt, "HIT", "DAE_G4PMTHIT_PATH_TEMPLATE");

        CopyToRemote(phl, evt, "g4pmthit");


    }
    {
        G4DAESensDet* sd = GetSensDet();

        G4DAEPmtHitList* phl = new G4DAEPmtHitList(1000);

        sd->PopulatePmtHitList(phl);  // from chroma HC into PmtHitList

        phl->Print("G4DAEChroma::PostProcessing from SD");

        //phl->Save(evt);  // standard PMTHIT template

        CopyToRemote(phl, evt, "pmthit");

    }
    


    DumpResults("G4DAEChroma::EndOfEvent");
    UpdateResults();

    G4DAEMetadata* results = GetResults();
    results->Print("#results");
    results->PrintLinks("#results_links");

    G4DAEDatabase* db = GetDatabase(); 
    if(db)
    {
        db->Insert(results, "tevent",  "BeginOfEvent,EndOfEvent" );
    }
    else
    {
        printf("G4DAEChroma::EndOfEvent db NULL \n");
    }
}



void G4DAEChroma::GPUProcessing(int evt)
{
    Start("CHROMA_PROCESS", 1);

    size_t TASK_G4CERENKOV_PROCESS_STEP        = FindTask("G4CERENKOV_PROCESS_STEP");
    size_t TASK_G4CERENKOV_PROCESS_PHOTON      = FindTask("G4CERENKOV_PROCESS_PHOTON");
    size_t TASK_G4SCINTILLATION_PROCESS_STEP   = FindTask("G4SCINTILLATION_PROCESS_STEP");
    size_t TASK_G4SCINTILLATION_PROCESS_PHOTON = FindTask("G4SCINTILLATION_PROCESS_PHOTON");

    if(TASK_G4CERENKOV_PROCESS_STEP)
    {
        Start(TASK_G4CERENKOV_PROCESS_STEP, 1);

        ProcessCerenkovSteps(evt);    
        Stop(TASK_G4CERENKOV_PROCESS_STEP, 1);
    }
    else
    {
        Skip(TASK_G4CERENKOV_PROCESS_STEP, 1);
    }


    if(TASK_G4SCINTILLATION_PROCESS_STEP)
    {
        Start(TASK_G4SCINTILLATION_PROCESS_STEP, 1);
        ProcessScintillationSteps(evt);    
        Stop(TASK_G4SCINTILLATION_PROCESS_STEP, 1);
    }
    else
    {
        Skip(TASK_G4SCINTILLATION_PROCESS_STEP, 1);
    }


    if(TASK_G4CERENKOV_PROCESS_PHOTON)
    {
        Start(TASK_G4CERENKOV_PROCESS_PHOTON, 1);
        ProcessCerenkovPhotons(evt);    
        Stop(TASK_G4CERENKOV_PROCESS_PHOTON, 1);
    }
    else
    {
        Skip(TASK_G4CERENKOV_PROCESS_PHOTON, 1); 
    }



    if(TASK_G4SCINTILLATION_PROCESS_PHOTON)
    {
        Start(TASK_G4SCINTILLATION_PROCESS_PHOTON, 1);
        ProcessScintillationPhotons(evt);    
        Stop(TASK_G4SCINTILLATION_PROCESS_PHOTON, 1);
    }
    else
    {
        Skip(TASK_G4SCINTILLATION_PROCESS_PHOTON, 1);
    }

    Stop("CHROMA_PROCESS", 1);
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


