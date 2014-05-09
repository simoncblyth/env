#include "ChromaPhotonList.hh"
#include "assert.h"
#include <iostream>

ChromaPhotonList::ChromaPhotonList() : TObject() {
}

ChromaPhotonList::~ChromaPhotonList() {

}

void ChromaPhotonList::Print(Option_t* option) const 
{
    std::cout <<  "ChromaPhotonList::Print " << option << " [" << x.size() << "]" << std::endl ;    
} 


#ifdef WITH_GEANT4
void ChromaPhotonList::AddPhoton(G4ThreeVector pos, G4ThreeVector mom, G4ThreeVector pol, float _t, float _wavelength, int _pmtid) 
{
    x.push_back(pos.x());
    y.push_back(pos.y());
    z.push_back(pos.z());
    px.push_back(mom.x());
    py.push_back(mom.y());
    pz.push_back(mom.z());
    polx.push_back(pol.x());
    poly.push_back(pol.y());
    polz.push_back(pol.z());
    t.push_back(_t);
    wavelength.push_back(_wavelength);
    pmtid.push_back(_pmtid);
}
#endif


void ChromaPhotonList::AddPhoton(float _x, float _y, float _z,  float _momx, float _momy, float _momz, float _polx, float _poly, float _polz, float _t, float _wavelength, int _pmtid) 
{
    x.push_back(_x);
    y.push_back(_y);
    z.push_back(_z);
    px.push_back(_momx);
    py.push_back(_momy);
    pz.push_back(_momz);
    polx.push_back(_polx);
    poly.push_back(_poly);
    polz.push_back(_polz);
    t.push_back(_t);
    wavelength.push_back(_wavelength);
    pmtid.push_back(_pmtid);
}


void ChromaPhotonList::ClearAll() 
{
    x.clear();
    y.clear();
    z.clear();
    px.clear();
    py.clear();
    pz.clear();
    polx.clear();
    poly.clear();
    polz.clear();
    t.clear();
    wavelength.clear();
    pmtid.clear();
}


 
// Build a ChromaPhotonList object from C arrays
void ChromaPhotonList::FromArrays(float* _x,    float* _y,    float* _z,
                  float* _px,   float* _py,   float* _pz,
                  float* _polx, float* _poly, float* _polz,
                  float* _t, float* _wavelength, int* _pmtid, int nphotons) 
{
    for (int i=0; i<nphotons; i++) { 
      x.push_back(_x[i]);
      y.push_back(_y[i]);
      z.push_back(_z[i]);
      px.push_back(_px[i]);
      py.push_back(_py[i]);
      pz.push_back(_pz[i]);
      polx.push_back(_polx[i]);
      poly.push_back(_poly[i]);
      polz.push_back(_polz[i]);
      t.push_back(_t[i]);
      wavelength.push_back(_wavelength[i]);
      pmtid.push_back(_pmtid[i]);
    }
}



#ifdef WITH_GEANT4
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
#endif

#ifdef WITH_GEANT4
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
#endif



