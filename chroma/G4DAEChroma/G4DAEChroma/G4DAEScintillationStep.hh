
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

       _charge, 
       _BetaInverse,
       _weight, 
       _MeanVelocity,

       _Pmin,  
       _Pmax,   
       _dp,    
       _maxCos,

       _maxSin2,
       _MeanNumberOfPhotons1,
       _MeanNumberOfPhotons2,
       _MeanNumberOfPhotonsMax,

       SIZE

    };

    int id ; // avoid empty 
};


#endif 


