#include "ChromaPhotonList.hh"
#include "assert.h"


ChromaPhotonList::ChromaPhotonList() : TObject() {
}

ChromaPhotonList::~ChromaPhotonList() {

}

void ChromaPhotonList::Print(Option_t* option) const 
{
    std::cout <<  "ChromaPhotonList::Print [" << x.size() << "]" << std::endl ;    
} 


void ChromaPhotonList::Details() const 
{
    std::cout <<  "ChromaPhotonList::Details [" << x.size() << "]" << std::endl ;

    G4ThreeVector pos ;
    G4ThreeVector mom ;
    G4ThreeVector pol ;
    float _t ;
    float _wavelength ;
    int _pmtid ;

    size_t index ; 
    for( index = 0 ; index < x.size() ; index++ )
    {
        GetPhoton( index , pos, mom, pol, _t, _wavelength, _pmtid );    
        G4cout << " index " << index << " pos " << pos << " mom " << mom << " pol " << pol << " _t " << _t << " _wavelength " << _wavelength << " _pmtid " << _pmtid << G4endl ; 
    }
} 

void ChromaPhotonList::GetPhoton(size_t index, G4ThreeVector& pos, G4ThreeVector& mom, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const
{
    assert( index < x.size() );

    pos.setX( x[index] );  
    pos.setY( y[index] );  
    pos.setZ( z[index] );  

    mom.setX( px[index] );  
    mom.setY( py[index] );  
    mom.setZ( pz[index] );  

    pol.setX( polx[index] );  
    pol.setY( poly[index] );  
    pol.setZ( polz[index] );  

    _t = t[index] ;
    _wavelength = wavelength[index] ;
    _pmtid = pmtid[index] ;

}




