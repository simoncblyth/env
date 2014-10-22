#include "G4DAEChroma/G4DAEDayabay.hh"

#include "Event/SimPmtHit.h"
#include "Conventions/Detectors.h"

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


G4DAEDayabay::G4DAEDayabay()
{
    DefineCollectionNames();
}

G4DAEDayabay::~G4DAEDayabay()
{
}

void G4DAEDayabay::DefineCollectionNames()
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


void G4DAEDayabay::CreateHitCollections( const char* sdname, G4HCofThisEvent* hce )
{
    m_hc.clear();

    int noc = hce->GetNumberOfCollections();

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
            G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,name.c_str());
            int hcid = G4SDManager::GetSDMpointer()->GetCollectionID(hc);
            hce->AddHitsCollection(hcid,hc);

            short int id = det.siteDetPackedData();
            m_hc[id] = hc;

        }       
    }
    cout << "G4DAEDayabay::CreateHitCollections "
         << " HCE " << hce
         << " SDN " << SensitiveDetectorName 
         << " add #collections  " << hce->GetNumberOfCollections() - noc  
         << " tot " << hce->GetNumberOfCollections()
         << endl; 
}


void G4DAEDayabay::CollectHit( const G4DAEHit& hit )
{
    int trackid = hit.trackid ; 
    DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();

    sphit->setSensDetId(hit.pmtid);
    sphit->setLocalPos(hit.lpos);
    sphit->setHitTime(hit.t);
    sphit->setPol(hit.lpol);
    sphit->setDir(hit.ldir);
    sphit->setWavelength(hit.wavelength);
    sphit->setType(0);
    sphit->setWeight(hit.weight);

    DayaBay::Detector det(hit.pmtid);
    short int sdid = det.siteDetPackedData();

    G4DhHitCollection* hc = m_hc[sdid];

    if (!hc) { 
        cout  << "G4DAEDayabay::CollectHit : WARNING hit with no hit collection. " 
              << " pmtid " << (void*)hit.pmtid
              << " det: " << setw(15) << det.detName()
              << " Storing to collectionName[0] " << collectionName[0]
              << endl; 
        sdid = 0;
        hc = m_hc[sdid];
    }

    cout << "G4DAEDayabay::CollectHit "
         << " pmtid : " << (void*)pmtid 
         << " from " << setw(15) << det.detName()
         << " sdid " <<  setw(5) << sdid 
         << " (void*)sdid " << (void*)sdid
         << " t " << sphit->hitTime()/CLHEP::ns << "[ns] " 
         << " pos " << sphit->localPos()/CLHEP::cm << "[cm] " 
         << " wav " << sphit->wavelength()/CLHEP::nm << "[nm]"
         << endl; 

    hc->insert(new G4DhHit(sphit,trackid));
}



