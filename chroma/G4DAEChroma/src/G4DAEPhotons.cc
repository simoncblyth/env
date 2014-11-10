#include "G4DAEChroma/G4DAEPhotons.hh"
#include "G4ThreeVector.hh"
#include <cstdlib>

void G4DAEPhotons::Transfer( G4DAEPhotons* dest , G4DAEPhotons* src )
{
   size_t nphoton = src->GetPhotonCount();

   G4ThreeVector pos ;
   G4ThreeVector mom ; 
   G4ThreeVector pol ; 
   float _t ; 
   float _wavelength ; 
   int _pmtid ;

   for(size_t index=0 ; index < nphoton ; ++index){
       src->GetPhoton(index, pos, mom, pol, _t, _wavelength, _pmtid);
       dest->AddPhoton(pos, mom, pol, _t, _wavelength, _pmtid);
   }   
}





