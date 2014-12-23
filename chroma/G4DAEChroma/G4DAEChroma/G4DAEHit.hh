
#ifndef G4DAEHIT_H
#define G4DAEHIT_H 1

#include "G4ThreeVector.hh"

class G4DAEArrayHolder ; 

#include <iostream>
#include <cstddef>

class G4AffineTransform;

struct G4DAEHit {

    enum {
       _gpos_x,
       _gpos_y,
       _gpos_z,

       _gdir_x,
       _gdir_y,
       _gdir_z,

       _gpol_x,
       _gpol_y,
       _gpol_z,

       _lpos_x,
       _lpos_y,
       _lpos_z,

       _ldir_x,
       _ldir_y,
       _ldir_z,

       _lpol_x,
       _lpol_y,
       _lpol_z,

       _t, 
       _wavelength,
       _weight,

       _pmtid,
       _photonid,
       _spare

    };

    // global
    G4ThreeVector gpos; 
    G4ThreeVector gdir; 
    G4ThreeVector gpol;

    // local 
    G4ThreeVector lpos; 
    G4ThreeVector ldir; 
    G4ThreeVector lpol; 

    float t  ; 
    float wavelength ; 
    float weight ;     

    int pmtid ; 
    int photonid ;
    int spare ;


    void Init(G4DAEArrayHolder* photons, std::size_t index);
    void Serialize(float* data);
    void InitFake( std::size_t pmtid_, std::size_t photonid_ );
    void LocalTransform(G4AffineTransform* trans);
    void Print(const char* msg="G4DAEHit::Print") const;

};



#endif 


