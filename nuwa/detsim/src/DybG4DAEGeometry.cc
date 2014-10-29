#include "DybG4DAEGeometry.h"

#include "GaudiKernel/ToolHandle.h"
#include "DetDesc/DetectorElement.h"
#include "G4DataHelpers/ITouchableToDetectorElement.h"

#include "G4TouchableHistory.hh"
#include <iostream>
using namespace std ; 

DybG4DAEGeometry::DybG4DAEGeometry(ITouchableToDetectorElement* t2de, const char* idParameter) : 
     G4DAEGeometry(), 
     m_idParameter(idParameter),
     //m_t2deName(t2deName), 
     m_t2de(t2de)
{
    //m_t2de = tool<ITouchableToDetectorElement>(m_t2deName);
}

DybG4DAEGeometry::~DybG4DAEGeometry()
{
}


std::size_t DybG4DAEGeometry::TouchableToIdentifier( const G4TouchableHistory& hist )
{
     // following NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPmtSensDet.cc DsPmtSensDet::ProcessHits

    const DetectorElement* de = this->SensDetElem(hist);
    if (!de) return 0;

    int pmtid = this->SensDetId(*de);

    return pmtid ; 
}



const DetectorElement* DybG4DAEGeometry::SensDetElem(const G4TouchableHistory& hist)
{
    const IDetectorElement* idetelem = 0;
    int steps=0;

    if (!hist.GetHistoryDepth()) {
        cout << "error() " << "DybG4DAEGeometry::SensDetElem given empty touchable history" << endreq;
        return 0;
    }   

    StatusCode sc = 
        m_t2de->GetBestDetectorElement(&hist,m_sensorStructures,idetelem,steps);
    if (sc.isFailure()) {      // verbose warning
        cout << "warning() " << "Failed to find detector element in:\n";
        for (size_t ind=0; ind<m_sensorStructures.size(); ++ind) {
            cout << "warning() " << "\t\t" << m_sensorStructures[ind] << "\n";
        }   
        cout << "warning() " << "\tfor touchable history:\n";
        for (int ind=0; ind < hist.GetHistoryDepth(); ++ind) {
            cout << "warning() " << "\t (" << ind << ") " 
                      << hist.GetVolume(ind)->GetName() << "\n";
        }   
        cout << "warning() " << endreq;
        return 0;
    }   
    
    return dynamic_cast<const DetectorElement*>(idetelem);
}

int  DybG4DAEGeometry::SensDetId(const DetectorElement& de) 
{
    const DetectorElement* detelem = &de;

    while (detelem) {
        if (detelem->params()->exists(m_idParameter)) {
            break;
        }   
        detelem = dynamic_cast<const DetectorElement*>(detelem->parentIDetectorElement());
    }   
    if (!detelem) {
        cout << "warning() " << "Could not get PMT detector element starting from " << endl;
        return 0;
    }   

    return detelem->params()->param<int>(m_idParameter);
}



