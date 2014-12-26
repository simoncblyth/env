#ifndef G4DAECERENKOVPHOTON_H
#define G4DAECERENKOVPHOTON_H 

#include "G4DAEChroma/G4DAEPhoton.hh"

class G4DAECerenkovPhoton : public G4DAEPhoton  {
    public:

    static const char* TMPL ;   // name of envvar containing path template 
    static const char* SHAPE ;  // numpy array itemshape eg "8,3" or "4,4" 
    static const char* KEY ;  

};


#endif 


