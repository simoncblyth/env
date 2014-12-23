#ifndef G4DAEPHOTON_H
#define G4DAEPHOTON_H 

#include "G4ThreeVector.hh"

class G4DAEArrayHolder ; 
class G4Track ; 

class G4DAEPhoton  {
    public:

    static const char* TMPL ;   // name of envvar containing path template 
    static const char* SHAPE ;  // numpy array itemshape eg "8,3" or "4,4" 
    static const char* KEY ;  

    static void Collect(G4DAEArrayHolder* photons, const G4Track* aPhoton );
    static void Collect(G4DAEArrayHolder* photons, const G4ThreeVector& pos, const G4ThreeVector& dir, const G4ThreeVector& pol, const float time, const float wavelength, const int pmtid=-1);
    static void Dump(G4DAEArrayHolder* photons, bool /*hit*/);
    static void Get( G4DAEArrayHolder* photons, std::size_t index , G4ThreeVector& pos, G4ThreeVector& dir, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid );

    enum {
       _post_x,
       _post_y,
       _post_z,
       _post_w,

       _dirw_x,
       _dirw_y,
       _dirw_z,
       _dirw_w,

       _polw_x,
       _polw_y,
       _polw_z,
       _polw_w,

       _flag_x,
       _flag_y,
       _flag_z,
       _flag_w
    };


};


#endif 


