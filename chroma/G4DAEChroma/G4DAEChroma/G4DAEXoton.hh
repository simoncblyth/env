#ifndef G4DAEXOTON_H
#define G4DAEXOTON_H 

#include "G4DAEChroma/G4DAEFoton.hh"

class G4DAEXoton : public G4DAEFoton  {
    public:

    static const char* TMPL ;   // name of envvar containing path template 
    static const char* SHAPE ;  // numpy array itemshape eg "8,3" or "4,4" 
    static const char* KEY ;  

};


#endif 


