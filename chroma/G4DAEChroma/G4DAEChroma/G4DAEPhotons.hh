#ifndef G4DAEPHOTONS_H
#define G4DAEPHOTONS_H

#include <G4ThreeVector.hh>
#include <cstdlib>
#include <string>

class G4DAEPhotonList ;
class G4DAEChromaPhotonList ;

#include "G4DAEChroma/G4DAESerializable.hh"

class G4DAEPhotons : public G4DAESerializable {

public:

  virtual ~G4DAEPhotons(){};

  virtual void AddPhoton(G4ThreeVector pos, G4ThreeVector mom, G4ThreeVector pol, float _t, float _wavelength, int _pmtid=-1) = 0 ;
  virtual void GetPhoton(size_t index, G4ThreeVector& pos, G4ThreeVector& mom, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const = 0 ; 
  virtual void Print() const = 0 ;
  virtual void Details(bool hit) const = 0 ;
  virtual std::size_t GetPhotonCount() const = 0;
  virtual std::string GetPhotonDigest() const = 0;
  virtual void ClearAllPhotons() = 0;

  static void Transfer( G4DAEPhotons* dest , G4DAEPhotons* src );
  static G4DAEPhotons* LoadPath( const char* path , const char* key="NPL");
  static G4DAEPhotons* Load(   const char* name , const char* key="NPL", const char* tmpl="DAE_PATH_TEMPLATE" );

  static void SavePath( G4DAEPhotonList* photons, const char* path , const char* key="NPL");
  static void SavePath( G4DAEChromaPhotonList* photons, const char* path , const char* key="CPL");
  static void Save( G4DAEPhotons* photons, const char* name, const char* key="NPL", const char* tmpl="DAE_PATH_TEMPLATE" );

  static bool HasExt(const char* path, const char* ext);
  static std::string SwapExt(const char* path, const char* aext, const char* bext);


};


#endif
