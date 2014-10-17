
#include "Event/SimPmtHit.h"
#include "Conventions/Detectors.h"

DetectorId::DetectorId_t detector_ids[] =
{  DetectorId::kAD1, DetectorId::kAD2, DetectorId::kAD3,
   DetectorId::kAD4, DetectorId::kIWS, DetectorId::kOWS,
   (DetectorId::DetectorId_t)-1 };

Site::Site_t site_ids[] =
{ Site::kDayaBay, Site::kLingAo, Site::kFar, (Site::Site_t)-1 };


using namespace std; 

int main()
{

   for (int isite=0; site_ids[isite] >= 0; ++isite) {
        Site::Site_t site = site_ids[isite];

        for (int idet=0; detector_ids[idet] >= 0; ++idet) {
            DetectorId::DetectorId_t detid = detector_ids[idet];

            DayaBay::Detector det(site,detid);

            if (det.bogus()) continue;

            string name=det.detName();
            cout << name << endl ;   
        }
    }
    

}



