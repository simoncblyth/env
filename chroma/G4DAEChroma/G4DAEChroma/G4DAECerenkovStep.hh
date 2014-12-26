#ifndef G4DAECERENKOVSTEP_H
#define G4DAECERENKOVSTEP_H 

// machinery to serialize the stack from DsG4Cerenkov::PostStepDoIt 

class G4DAECerenkovStep {
    public:

    static const char* TMPL ;   // name of envvar containing path template 
    static const char* SHAPE ;  // numpy array itemshape eg "8,3" or "4,4" 
    static const char* KEY ;  

    enum {

       _Id,                      //  0
       _ParentID,
       _Material,
       _NumPhotons,
      
       _x0_x,                    //  1
       _x0_y,
       _x0_z,
       _t0,

       _DeltaPosition_x,         // 2
       _DeltaPosition_y,
       _DeltaPosition_z,
       _step_length,

       _code,                    // 3
       _charge, 
       _weight, 
       _MeanVelocity,

       _BetaInverse,             //  4
       _Pmin,  
       _Pmax,   
       _maxCos,

       _maxSin2,                 // 5
       _MeanNumberOfPhotons1,
       _MeanNumberOfPhotons2,
       _BialkaliMaterialIndex,

       SIZE

    };

};


#endif 


