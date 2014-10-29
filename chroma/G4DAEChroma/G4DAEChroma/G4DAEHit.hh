
#ifndef G4DAEHIT_H
#define G4DAEHIT_H 1

#include "G4ThreeVector.hh"
#include <iostream>
#include <cstddef>

class G4AffineTransform;
class ChromaPhotonList;

struct G4DAEHit {

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
    int trackid ;


    void Init(ChromaPhotonList* cpl, std::size_t index);
    void InitFake( std::size_t sensor_id, std::size_t track_id );
    void LocalTransform(G4AffineTransform* trans);
    void Print();

};



#endif 


