#ifndef G4DAEPHOTONLIST_H
#define G4DAEPHOTONLIST_H

#include "G4DAEChroma/G4DAEPhoton.hh"
#include "G4DAEChroma/G4DAEList.hh"

typedef G4DAEList<G4DAEPhoton> G4DAEPhotonList ;



/*
#include <G4ThreeVector.hh>

class G4DAEPhotonList : public G4DAEList<G4DAEPhoton> {
public:
  G4DAEPhotonList(G4DAEArray* array);
  G4DAEPhotonList( std::size_t itemcapacity = 0, float* data = NULL);
  virtual ~G4DAEPhotonList();

  void AddPhoton(G4ThreeVector pos, G4ThreeVector mom, G4ThreeVector pol, float _t, float _wavelength, int _pmtid=-1); 
  void GetPhoton(size_t index, G4ThreeVector& pos, G4ThreeVector& mom, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const;

  void Print(const char* msg="G4DAEPhotonList::Print") const ; 
  void Details(bool hit) const ;

};
*/


#endif

