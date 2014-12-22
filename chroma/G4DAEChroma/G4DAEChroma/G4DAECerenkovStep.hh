#ifndef G4DAECERENKOVSTEP_H
#define G4DAECERENKOVSTEP_H 

// machinery to serialize the stack from DsG4Cerenkov::PostStepDoIt 

class G4DAECerenkovStep {
    public:

    static const char* TMPL ;   // name of envvar containing path template 
    static const char* SHAPE ;  // numpy array itemshape eg "8,3" or "4,4" 
    static const char* KEY ;  

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

       _BetaInverse,
       _Pmin,  
       _Pmax,   
       _maxCos,

       _maxSin2,
       _MeanNumberOfPhotons1,
       _MeanNumberOfPhotons2,
       _MeanNumberOfPhotonsMax,

       SIZE

    };

};


#endif 


