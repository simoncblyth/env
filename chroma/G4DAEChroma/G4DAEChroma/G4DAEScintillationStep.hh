
#ifndef G4DAESCINTILLATIONSTEP_H
#define G4DAESCINTILLATIONSTEP_H 

#include "G4ThreeVector.hh"

// machinery to serialize the stack from DsChromaG4Scintillation::PostStepDoIt 

struct G4DAEScintillationStep {

    enum {

       _Id, 
       _ParentID,
       _Material,
       _NumPhotons,
      
       _x0_x,
       _x0_y,
       _x0_z,
       _t0,

       _DeltaPosition_x,
       _DeltaPosition_y,
       _DeltaPosition_z,
       _step_length,

       _code,
       _charge, 
       _weight, 
       _MeanVelocity,

       _scnt,  
       _slowerRatio,   
       _slowTimeConstant,    
       _slowerTimeConstant,

       _ScintillationTime,
       _ScintillationIntegralMax,
       _Spare1,
       _Spare2,

       SIZE

    };

    int id ; // avoid empty 
};


#endif 


