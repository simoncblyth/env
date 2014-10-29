#ifndef DYBG4DAEGEOMETRY_H
#define DYBG4DAEGEOMETRY_H 1

#include "GiGa/GiGaBase.h"
#include "G4DAEChroma/G4DAEGeometry.hh"

#include <cstddef>
#include <string>
#include <vector>

class ITouchableToDetectorElement ;
class DetectorElement ;
class G4TouchableHistory ;

// 
// Dayabay specialization of G4DAEGeometry
// that provides TouchableToIdentifier method
//
// mostly duplicating code from  
//    NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPmtSensDet.cc
//
//

class DybG4DAEGeometry : public G4DAEGeometry {

public:
    DybG4DAEGeometry(ITouchableToDetectorElement* t2de, const char* idParameter);
    virtual ~DybG4DAEGeometry();
   
    virtual std::size_t TouchableToIdentifier( const G4TouchableHistory& hist );

private:


    /// CathodeLogicalVolumes : name of logical volumes in which this
    /// sensitive detector is operating.
    std::vector<std::string> m_cathodeLogVols;

    /// SensorStructures : names of paths in TDS in which to search
    /// for sensor detector elements using this sensitive detector.
    std::vector<std::string> m_sensorStructures;

    /// PackedIdParameterName : name of user paramater of the counted
    /// detector element which holds the packed, globally unique PMT
    /// ID.
    std::string m_idParameter;

    /// TouchableToDetelem : the ITouchableToDetectorElement to use to
    /// resolve sensor ID.
    std::string m_t2deName;

    ITouchableToDetectorElement* m_t2de;


    // lookup DE from touchable history
    const DetectorElement* SensDetElem(const G4TouchableHistory& hist);
    // lookup ID from DE or daughter DE
    int SensDetId(const DetectorElement& de);



 
};

#endif



