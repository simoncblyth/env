#pragma once

#include <glm/fwd.hpp>
#include <string>

class GCSG ; 
class G4VSolid;
class Opticks ; 

//
// CMaker is a constitent of CTestDetector used
// to convert GCSG geometry into G4 geometry in 
// G4VPhysicalVolume* CTestDetector::Construct() 
//

#include "CFG4_API_EXPORT.hh"

class CFG4_API CMaker {
    public:
        static std::string PVName(const char* shapename);
        static std::string LVName(const char* shapename);
    public:
        CMaker(Opticks* cache, int verbosity=0);
    public:
        G4VSolid* makeSolid(char shapecode, const glm::vec4& param);
        G4VSolid* makeBox(const glm::vec4& param);
        G4VSolid* makeSphere(const glm::vec4& param);
    public:
        G4VSolid* makeSolid(GCSG* csg, unsigned int i);
    private:
        Opticks* m_cache ; 
        int      m_verbosity ; 

};


