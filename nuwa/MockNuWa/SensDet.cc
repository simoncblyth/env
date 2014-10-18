
#include "SensDet.h"


#include "Event/SimPmtHit.h"
#include "Conventions/Detectors.h"

#include "G4SDManager.hh"


using namespace std; 

DetectorId::DetectorId_t detector_ids[] =
{  DetectorId::kAD1, DetectorId::kAD2, DetectorId::kAD3,
   DetectorId::kAD4, DetectorId::kIWS, DetectorId::kOWS,
   (DetectorId::DetectorId_t)-1 };

Site::Site_t site_ids[] =
{ Site::kDayaBay, Site::kLingAo, Site::kFar, (Site::Site_t)-1 };


SensDet::SensDet(const std::string& name) : G4VSensitiveDetector(name)
{
}

SensDet::~SensDet()
{
}


int SensDet::initialize()
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


    return 0 ;
}



void SensDet::Initialize( G4HCofThisEvent* hce )
{

    m_hc.clear();

    cout << "make G4DhHitCollection SensitiveDetectorName " << SensitiveDetectorName << " collectionName[0] " << collectionName[0] << endl ; 

    //G4THitsCollection<G4DhHit>
    G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,collectionName[0]);

    m_hc[0] = hc;
    int hcid = G4SDManager::GetSDMpointer()->GetCollectionID(hc);
    cout << " hc " << (void*)hc << " hcid " << hcid << endl ;

    hce->AddHitsCollection(hcid,hc);

    for (int isite=0; site_ids[isite] >= 0; ++isite) {
        for (int idet=0; detector_ids[idet] >= 0; ++idet) {
            DayaBay::Detector det(site_ids[isite],detector_ids[idet]);

            if (det.bogus()) continue;

            string name=det.detName();
            G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,name.c_str());
            short int id = det.siteDetPackedData();
            m_hc[id] = hc;

            int hcid = G4SDManager::GetSDMpointer()->GetCollectionID(hc);
            hce->AddHitsCollection(hcid,hc);
            cout  << "Add hit collection with hcid=" << hcid << ", cached ID=" 
                    << (void*)id 
                    << " name= \"" << SensitiveDetectorName << "/" << name << "\"" 
                    << " hc= " << hc  
                    << endl; 
        }       
    }

    cout << "SensDet Initialize, made "
           << hce->GetNumberOfCollections() << " collections"
           << endl; 
    

}


    
bool SensDet::ProcessHits(G4Step* step, G4TouchableHistory* history)
{

   int pmtid = 0x1010101 ;
   int trackid = 101 ;

   DayaBay::Detector detector(pmtid);

   if (detector.detectorId() == DetectorId::kIWS || detector.detectorId() == DetectorId::kOWS) {
      cout << "IWS/OWS" << endl ; 
   } else {
      cout << "IWS/OWS not" << endl ; 
   } 

   DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
   sphit->setSensDetId(pmtid); 
   this->StoreHit(sphit,trackid);

   return 1; 
}




void SensDet::StoreHit(DayaBay::SimPmtHit* hit, int trackid)
{
   int did = hit->sensDetId();
   DayaBay::Detector det(did);
   short int sdid = det.siteDetPackedData();
     
   G4DhHitCollection* hc = m_hc[sdid];
   if (!hc) {
            cout << "Got hit with no hit collection.  ID = " << (void*)did
                 << " which is detector: \"" << DayaBay::Detector(did).detName()
                 << "\". Storing to the " << collectionName[0] << " collection"
                 << endl;
             sdid = 0;
             hc = m_hc[sdid];
    }

#if 1
    cout      << "Storing hit PMT: " << (void*)did 
              << " from " << DayaBay::Detector(did).detName()
              << " in hc #"<<  sdid << " = "
              << hit->hitTime()/CLHEP::ns << "[ns] " 
              << hit->localPos()/CLHEP::cm << "[cm] " 
              << hit->wavelength()/CLHEP::nm << "[nm]"
              << endl; 
#endif


    hc->insert(new G4DhHit(hit,trackid));
}

void SensDet::EndOfEvent( G4HCofThisEvent* hce ) 
{
    cout << "Cache has " << m_hc.size() << " collections" << endl ; 

    int ncols = hce->GetNumberOfCollections();
    cout << "SensDet EndOfEvent " << ncols << " collections.";

    int tothits = 0;
    for (int ind=0; ind<ncols; ++ind) {
      G4VHitsCollection* hc = hce->GetHC(ind);
      if ( hc->GetSize() > 0)
      {
          if ( tothits == 0) cout << endl; 
          cout << ind << ": " 
               << hc->GetSDname() << "//" << hc->GetName() << " has " 
               << hc->GetSize() << " hits" << endl; 
      }
      tothits += hc->GetSize() ;
    }
    if ( tothits == 0 ) cout << " No hits found in " << ncols << " collections."  << endl;


}

