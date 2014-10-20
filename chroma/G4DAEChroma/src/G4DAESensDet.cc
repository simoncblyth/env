
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "Chroma/ChromaPhotonList.hh"  

#include "G4SDManager.hh"


using namespace std; 

#ifdef G4DAE_DAYABAY

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

#endif


G4DAESensDet::G4DAESensDet(const std::string& name) : G4VSensitiveDetector(name), m_geometry(0)
{
}

G4DAESensDet::~G4DAESensDet()
{
}


int G4DAESensDet::initialize()
{
   DefineCollectionNames();
   return 0 ; 
}
void G4DAESensDet::Initialize( G4HCofThisEvent* hce )
{
    CreateHitCollections( hce );
}
bool G4DAESensDet::ProcessHits(G4Step* step, G4TouchableHistory* history)
{
    //assert(0);
   int pmtid = 0x1010101 ;
   int trackid = 101 ;
#ifdef G4DAE_DAYABAY
   DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
   sphit->setSensDetId(pmtid); 
   this->StoreHit( sphit, trackid);
#endif
   return true ; 
}






void G4DAESensDet::EndOfEvent( G4HCofThisEvent* hce ) 
{
    DumpStatistics(hce);
}


void G4DAESensDet::SetGeometry(G4DAEGeometry* geo){
   m_geometry = geo ; 
}

G4DAEGeometry* G4DAESensDet::GetGeometry(){
   return m_geometry ;
}




void G4DAESensDet::DefineCollectionNames()
{
#ifdef G4DAE_DAYABAY
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
#endif


}

void G4DAESensDet::CreateHitCollections( G4HCofThisEvent* hce )
{
    m_hc.clear();

#ifdef G4DAE_DAYABAY

    cout << "G4DAESensDet::CreateHitCollections SensitiveDetectorName " << SensitiveDetectorName << endl ; 
    cout << "G4DAESensDet::CreateHitCollections collectionName[0] " << collectionName[0] << endl ; 
    //G4THitsCollection<G4DhHit>
    G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,collectionName[0]);

    m_hc[0] = hc;
    int hcid = G4SDManager::GetSDMpointer()->GetCollectionID(hc);
    //cout << " hc " << (void*)hc << " hcid " << hcid << endl ;

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

            /*
            cout  << "Add hit collection with hcid=" << hcid << ", cached ID=" 
                    << (void*)id 
                    << " name= \"" << SensitiveDetectorName << "/" << name << "\"" 
                    << " hc= " << hc  
                    << endl; 
            */ 
        }       
    }

#endif

    cout << "G4DAESensDet::CreateHitCollections : hce now has  "
           << hce->GetNumberOfCollections() << " collections"
           << endl; 
    

}


    






void G4DAESensDet::CollectHits(ChromaPhotonList* cpl)
{
    cout << "G4DAESensDet::CollectHits " <<  endl ;   
    cpl->Print();
    std::size_t size = cpl->GetSize(); 
    for( std::size_t index = 0 ; index < size ; index++ )
    {
          CollectOneHit( cpl,  index );
    }   
}


void G4DAESensDet::CollectOneHit( ChromaPhotonList* cpl , std::size_t index )
{
    cout << "G4DAESensDet::CollectOneHit " <<  index <<  endl ;   

    G4ThreeVector gpos ; 
    G4ThreeVector gdir ; 
    G4ThreeVector gpol ;
    float t(0) ; 
    float wavelength(0) ; 
    int pmtid(0) ; 

    int trackid(0) ;
    int volumeindex(0) ; 
    float weight(1) ;     

    cpl->GetPhoton( index, gpos, gdir, gpol, t, wavelength, pmtid );    

    G4AffineTransform& trans = m_geometry->GetNodeTransform(volumeindex) ;

    G4ThreeVector pos = trans.TransformPoint(gpos);
    G4ThreeVector pol = trans.TransformAxis(gpol);
    G4ThreeVector dir = trans.TransformAxis(gdir);


#ifdef G4DAE_DAYABAY
    DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();

    sphit->setSensDetId(pmtid);
    sphit->setLocalPos(pos);
    sphit->setHitTime(t);
    sphit->setPol(pol);
    sphit->setDir(dir);
    sphit->setWavelength(wavelength);
    sphit->setType(0);
    sphit->setWeight(weight);

    this->StoreHit( sphit, trackid );
#endif
}



#ifdef G4DAE_DAYABAY
void G4DAESensDet::StoreHit(DayaBay::SimPmtHit* hit, int trackid)
{
    int did = hit->sensDetId();
    DayaBay::Detector det(did);
    short int sdid = det.siteDetPackedData();

    G4DhHitCollection* hc = m_hc[sdid];
    if (!hc) { 
        cout  << "Got hit with no hit collection.  ID = " << (void*)did
              << " which is detector: \"" << DayaBay::Detector(did).detName()
              << "\". Storing to the " << collectionName[0] << " collection"
              << endl; 
        sdid = 0;
        hc = m_hc[sdid];
    }

#if 1
    cout << "Storing hit PMT: " << (void*)did 
         << " from " << DayaBay::Detector(did).detName()
         << " in hc #"<<  sdid << " = "
         << hit->hitTime()/CLHEP::ns << "[ns] " 
         << hit->localPos()/CLHEP::cm << "[cm] " 
         << hit->wavelength()/CLHEP::nm << "[nm]"
         << endl; 
#endif

    hc->insert(new G4DhHit(hit,trackid));
}
#else
void G4DAESensDet::StoreHit(void* hit, int trackid)
{
}
#endif








void G4DAESensDet::AddSomeFakeHits()
{
    int myints[] = { 
                   0x1010101,
                   0x2010101,
                   0x4010101,
                 };  

    vector<int> pmtids( myints, myints + sizeof(myints) / sizeof(int) );
    for (vector<int>::iterator it = pmtids.begin(); it != pmtids.end(); ++it)
    {   
        int trackid = 1 ; 

#ifdef G4DAE_DAYABAY
        DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
        sphit->setSensDetId(*it); 
        StoreHit( sphit, trackid );
#endif



    }   
}








void G4DAESensDet::DumpStatistics( G4HCofThisEvent* hce ) 
{
    cout << "G4DAESensDet::DumpStatistics HCE Cache has " << m_hc.size() << " collections" << endl ; 

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



