#ifndef G4DAEPHOTONLIST_H
#define G4DAEPHOTONLIST_H

#include <vector>
#include <string>

#include <G4ThreeVector.hh>

#include "G4DAEChroma/G4DAEArray.hh"

class G4DAEPhotonList : public G4DAEArray {

public:

  G4DAEPhotonList( std::size_t itemcapacity = 0, float* data = NULL);
  virtual ~G4DAEPhotonList();

  void AddPhoton(G4ThreeVector pos, G4ThreeVector mom, G4ThreeVector pol, float _t, float _wavelength, int _pmtid=-1); 
  void GetPhoton(size_t index, G4ThreeVector& pos, G4ThreeVector& mom, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const;

  static std::string GetPath( const char* evt="dummy" , const char* tmpl="DAE_PATH_TEMPLATE_NPY");   
  static G4DAEPhotonList* Load(const char* evt="1", const char* key="GPL", const char* tmpl="DAE_PATH_TEMPLATE_NPY" );
  void Save(const char* evt="dummy", const char* key="GPL", const char* tmpl="DAE_PATH_TEMPLATE_NPY" );

  void Print() const ; 
  void Details(bool hit) const ;


};

#endif

