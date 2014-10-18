
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
            cout << name << endl ;   
        }
    }
    return 0 ;
}



void SensDet::Initialize( G4HCofThisEvent* hce )
{

    m_hc.clear();

    G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,collectionName[0]);
    m_hc[0] = hc;
    int hcid = G4SDManager::GetSDMpointer()->GetCollectionID(hc);
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
                    << endl; 
        }       
    }

    cout << "DsPmtSensDet Initialize, made "
           << hce->GetNumberOfCollections() << " collections"
           << endl; 
    

}

void SensDet::EndOfEvent( G4HCofThisEvent* HCE ) 
{
}
    
bool SensDet::ProcessHits(G4Step* step, G4TouchableHistory* history)
{
   return 1; 
}



