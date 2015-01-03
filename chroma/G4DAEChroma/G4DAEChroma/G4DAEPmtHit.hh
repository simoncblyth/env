#ifndef G4DAEPMTHIT_H
#define G4DAEPMTHIT_H 

class G4DAEPmtHit {
    public:

    static const char* TMPL ;   // name of envvar containing path template 
    static const char* SHAPE ;  // numpy array itemshape eg "8,3" or "4,4" 
    static const char* KEY ;  

    enum {
       _localPos_x,      //  0
       _localPos_y,
       _localPos_z,
       _hitTime,

       _dir_x,           // 1
       _dir_y,
       _dir_z,
       _wavelength,

       _pol_x,           // 2
       _pol_y, 
       _pol_z,
       _weight,

       _trackid,         // 3
       _aux1, 
       _aux2,
       _pmtid,

       SIZE
    };

};


#endif 


