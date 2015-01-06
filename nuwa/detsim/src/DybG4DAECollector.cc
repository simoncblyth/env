#include "DybG4DAECollector.h"

#include "G4HCofThisEvent.hh"
#include "G4SDManager.hh"

#include "Event/SimPmtHit.h"
#include "Conventions/Detectors.h"

#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAEPmtHitList.hh"
#include "G4DAEChroma/G4DAECommon.hh"


using namespace std ; 


DetectorId::DetectorId_t detector_ids[] = {              
                      DetectorId::kAD1, 
                      DetectorId::kAD2, 
                      DetectorId::kAD3,
                      DetectorId::kAD4, 
                      DetectorId::kIWS, 
                      DetectorId::kOWS,
          (DetectorId::DetectorId_t)-1 };

Site::Site_t site_ids[] = { 
                     Site::kDayaBay, 
                     Site::kLingAo, 
                     Site::kFar, 
                     (Site::Site_t)-1 };


DybG4DAECollector::DybG4DAECollector()
{
    DefineCollectionNames(collectionName);
}

DybG4DAECollector::~DybG4DAECollector()
{
}

void DybG4DAECollector::DefineCollectionNames(G4CollectionNameVector& collectionName)
{
   collectionName.insert("unknown");
   for (int isite=0; site_ids[isite] >= 0; ++isite) {
        Site::Site_t site = site_ids[isite];

        for (int idet=0; detector_ids[idet] >= 0; ++idet) {
            DetectorId::DetectorId_t detid = detector_ids[idet];

            DayaBay::Detector det(site,detid);

            if (det.bogus()) continue;

            string name=det.detName();
            collectionName.insert(name.c_str());
            //cout << "insert collectionName " << name << endl ;   
        }
    }
}



void DybG4DAECollector::CreateHitCollections( const std::string& sdname, G4HCofThisEvent* hce )
{
   /*
   When target is empty this is invoked from 

           void G4DAESensDet::Initialize( G4HCofThisEvent* hce )

   note that the created collections are added to the hce 
   */ 

    m_hc.clear();

#ifdef VERBOSE
    int noc = hce->GetNumberOfCollections();
#endif

    //G4THitsCollection<G4DhHit>
    G4DhHitCollection* hc = new G4DhHitCollection(sdname ,collectionName[0]);
    m_hc[0] = hc;
    int hcid = G4SDManager::GetSDMpointer()->GetCollectionID(hc);

    hce->AddHitsCollection(hcid,hc);

    for (int isite=0; site_ids[isite] >= 0; ++isite) {
        for (int idet=0; detector_ids[idet] >= 0; ++idet) {
            DayaBay::Detector det(site_ids[isite],detector_ids[idet]);
            if (det.bogus()) continue;

            string name=det.detName();
            G4DhHitCollection* hc = new G4DhHitCollection(sdname,name.c_str());
            int hcid = G4SDManager::GetSDMpointer()->GetCollectionID(hc);
            hce->AddHitsCollection(hcid,hc);

            short int id = det.siteDetPackedData();
            m_hc[id] = hc;

        }       
    }

#ifdef VERBOSE
    cout << "DybG4DAECollector::CreateHitCollections "
         << " HCE " << hce
         << " SDN " << sdname
         << " add #collections  " << hce->GetNumberOfCollections() - noc  
         << " tot " << hce->GetNumberOfCollections()
         << endl; 
#endif

}

void DybG4DAECollector::StealHitCollections(const std::string& target,  G4HCofThisEvent* HCE)
{
   /*
   Invoked from 
           void G4DAESensDet::Initialize( G4HCofThisEvent* hce )

   Summary: this steals HCE hit collection pointers of target SD

   For entries in HCtable with SDname matching the argument, 
   obtain hcid and corresponding HC. 
   Cache HC pointers into m_hc keyed by site-detector short int, 
   obtained by DayaBay::Detector interpretation of the collection name.   

   NB this relies on the `G4DAETrojanSensDet::Initialize( G4HCofThisEvent* hce )`
   being called after that of the target SD otherwise will fail to access HC.

   As a result of this access to targetted hit collections of the event
   hits can be added outside of the normal ProcessHits machinery using 
   hit collection methods provided by the `G4DAESensDet` base class.


   When combining normal G4 operations with GPU hits for 
   same event comparisons, it would be better not to do this.  Instead
   craete a separate SensDet 

   */ 

   m_hc.clear();
   G4SDManager* SDMan = G4SDManager::GetSDMpointer();

   G4HCtable* hct = SDMan->GetHCtable();
   for(G4int i=0 ; i < hct->entries() ; i++ )
   {
      string sdName = hct->GetSDname(i);  
      string colName = hct->GetHCname(i);  

      if(sdName != target) continue ;

      G4String query = sdName + "/" + colName ; 

      int hcid = hct->GetCollectionID(query);

      G4DhHitCollection* hc = (G4DhHitCollection*)HCE->GetHC(hcid); 

      DayaBay::Detector det(colName);
      if(det.bogus() && det != 0x0) cout << "DybG4DAECollector::StealHitCollections : WARNING bogus det " << det << endl ;
      //if(det.bogus()) continue ;
      short int detid = det.siteDetPackedData();

      if(m_hc.find(detid) != m_hc.end()) cout << "G4DAETrojanSensDet::StealHitCollections : WARNING : replacing hitcache entry with key " << detid << endl ;
      m_hc[detid] = hc ;

   } 

#ifdef VERBOSE
   cout << "DybG4DAECollector::StealHitCollections "
        << " HCE " << HCE
        << " target [" << target << "]"
        << " #col " << m_hc.size()
        << endl ; 
#endif
}


void DybG4DAECollector::Collect( const G4DAEHit& hit )
{
    int trackid = hit.photonid ; 
    DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();

    sphit->setSensDetId(hit.pmtid);
    sphit->setLocalPos(hit.lpos);
    sphit->setHitTime(hit.t);
    sphit->setPol(hit.lpol);
    sphit->setDir(hit.ldir);
    sphit->setWavelength(hit.wavelength);
    sphit->setType(0);
    sphit->setWeight(hit.weight);

    // pick hit site-det collection to insert into 
    DayaBay::Detector det(hit.pmtid);
    short int sdid = det.siteDetPackedData();

    G4DhHitCollection* hc = m_hc[sdid];

    if (!hc) 
    { 
        cout  << "DybG4DAECollector::CollectHit : WARNING hit with no hit collection. " 
              << " pmtid " << (void*)hit.pmtid
              << " sdid " << setw(5) << sdid
              << " (void*)sdid " << (void*)sdid
              << " det: " << setw(15) << det.detName()
              << " Storing to collectionName[0] " << collectionName[0]
              << endl; 
        sdid = 0;
        hc = m_hc[sdid];
    }

#ifdef VERBOSE
    cout << "DybG4DAECollector::CollectHit "
         << " hc : " << (void*)hc 
         << " pmtid : " << (void*)hit.pmtid 
         << " from " << setw(15) << det.detName()
         << " sdid " <<  setw(5) << sdid 
         << " (void*)sdid " << (void*)sdid
         << " t " << sphit->hitTime()/CLHEP::ns << "[ns] " 
         << " pos " << sphit->localPos()/CLHEP::cm << "[cm] " 
         << " wav " << sphit->wavelength()/CLHEP::nm << "[nm]"
         << endl; 
#endif


    if(hc == NULL)
    {
        cout << "DybG4DAECollector::CollectHit NULL hc cannot insert " << endl ; 
        DumpLocalHitCache();
    }
    else
    {
        hc->insert(new G4DhHit(sphit,trackid));
    }
}


void DybG4DAECollector::DumpLocalHitCache()
{
    cout << "DybG4DAECollector::DumpLocalHitCache m_hc size " << m_hc.size() << endl ; 
    for( LocalHitCache::iterator it=m_hc.begin() ; it != m_hc.end() ; it++ )
    {
         short int hcid = it->first ;
         G4DhHitCollection* hc = it->second ; 
         if(hc->GetSize() == 0) continue;

         cout 
             << " hcid " << hcid 
             << " hc " << hc
             << " size " << hc->GetSize() 
             << endl ; 

         DumpLocalHitCollection(hc); 
    }
}

void DybG4DAECollector::DumpLocalHitCollection(G4DhHitCollection* hc)
{

    //G4THitsCollection<G4DhHit>
    size_t size = hc->GetSize(); 
    for(size_t index = 0 ; index < size ; ++index )
    {
         G4DhHit* hit = dynamic_cast<G4DhHit*>(hc->GetHit(index));
         cout << " index " << index << " hit " << hit << endl ;  

         int trackid = hit->trackId();
         DayaBay::SimPmtHit* sphit = dynamic_cast<DayaBay::SimPmtHit*>(hit->get());

         cout << " trackid " << trackid 
              << " sphit " << sphit 
              << endl ;  

         const CLHEP::Hep3Vector& localPos = sphit->localPos(); 
         cout << "localPos " << localPos << endl ; 
    }
}




void DybG4DAECollector::PopulatePmtHitList(G4DAEPmtHitList* phl)
{
    cout << "DybG4DAECollector::PopulatePmtHitList" << endl; 

    for( LocalHitCache::iterator it=m_hc.begin() ; it != m_hc.end() ; it++ )
    {
         //short int hcid = it->first ;
         G4DhHitCollection* hc = it->second ; 

         size_t size = hc->GetSize(); 
         if(size == 0) continue;

         for(size_t index = 0 ; index < size ; ++index )
         {
             G4DhHit* hit = dynamic_cast<G4DhHit*>(hc->GetHit(index));
             DayaBay::SimPmtHit* sphit = dynamic_cast<DayaBay::SimPmtHit*>(hit->get());

             const CLHEP::Hep3Vector& localPos = sphit->localPos(); 
             double hitTime = sphit->hitTime();

             const CLHEP::Hep3Vector& dir = sphit->dir(); 
             double wavelength = sphit->wavelength();

             const CLHEP::Hep3Vector& pol = sphit->pol(); 
             float weight = sphit->weight();

             int trackid = hit->trackId();
             int type    = sphit->type();   // unused?
             int sensDetId = sphit->sensDetId();

             float* ph = phl->GetNextPointer();     

             ph[G4DAEPmtHit::_localPos_x] = localPos.x() ;
             ph[G4DAEPmtHit::_localPos_y] = localPos.y() ;
             ph[G4DAEPmtHit::_localPos_z] = localPos.z() ;
             ph[G4DAEPmtHit::_hitTime]    = hitTime  ;

             ph[G4DAEPmtHit::_dir_x] = dir.x() ;
             ph[G4DAEPmtHit::_dir_y] = dir.y() ;
             ph[G4DAEPmtHit::_dir_z] = dir.z() ;
             ph[G4DAEPmtHit::_wavelength] = wavelength ;

             ph[G4DAEPmtHit::_pol_x] = pol.x() ;
             ph[G4DAEPmtHit::_pol_y] = pol.y() ;
             ph[G4DAEPmtHit::_pol_z] = pol.z() ;
             ph[G4DAEPmtHit::_weight] = weight ;

             uif_t uifd[4] ; 
             uifd[0].i = trackid ;
             uifd[1].i = type ;  
             uifd[2].i = 0 ;
             uifd[3].i = sensDetId  ; 

             ph[G4DAEPmtHit::_trackid] = uifd[0].f ;
             ph[G4DAEPmtHit::_aux1]    = uifd[1].f ;
             ph[G4DAEPmtHit::_aux2]    = uifd[2].f ;
             ph[G4DAEPmtHit::_pmtid]   = uifd[3].f ;


         }
    }
}












