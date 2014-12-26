#ifndef G4DAESCINTILLATIONPHOTON_H
#define G4DAESCINTILLATIONPHOTON_H 

#include "G4DAEChroma/G4DAEPhoton.hh"

class G4DAEScintillationPhoton : public G4DAEPhoton {
    public:

    static const char* TMPL ;   // name of envvar containing path template 
    static const char* SHAPE ;  // numpy array itemshape eg "8,3" or "4,4" 
    static const char* KEY ;  

};


#endif 


