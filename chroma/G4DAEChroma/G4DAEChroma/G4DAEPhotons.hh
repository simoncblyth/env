#ifndef G4DAEPHOTONS_H
#define G4DAEPHOTONS_H

#include <G4ThreeVector.hh>
#include <cstdlib>
#include <string>

class G4DAEChromaPhotonList ;
class G4DAEPhotonListOld ;

#include "G4DAEChroma/G4DAESerializable.hh"

class G4DAEPhotons : public G4DAESerializable {

  static const char* TMPL ;
  static const char* KEY ;

public:

  virtual ~G4DAEPhotons(){};

  virtual void AddPhoton(G4ThreeVector pos, G4ThreeVector mom, G4ThreeVector pol, float _t, float _wavelength, int _pmtid=-1) = 0 ;
  virtual void GetPhoton(size_t index, G4ThreeVector& pos, G4ThreeVector& mom, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const = 0 ; 
  virtual void Print(const char* msg="G4DAEPhotons::Print") const = 0 ;
  virtual void Details(bool hit) const = 0 ;

  virtual std::size_t GetCount() const = 0;
  virtual std::string GetDigest() const = 0;
  virtual void ClearAll() = 0;
  virtual G4DAEPhotons* Slice(int a, int b) = 0;


  static void Transfer( G4DAEPhotons* dest , G4DAEPhotons* src, std::size_t a=0, std::size_t b=0 );
  static G4DAEPhotons* LoadPath( const char* path , const char* key=KEY);
  static G4DAEPhotons* Load(   const char* name , const char* key=KEY, const char* tmpl=TMPL );

  static void SavePath( G4DAEPhotonListOld* photons, const char* path , const char* key=KEY );
#ifdef G4DAECHROMA_WITH_CPL
  static void SavePath( G4DAEChromaPhotonList* photons, const char* path , const char* key="CPL");
#endif
  static void Save( G4DAEPhotons* photons, const char* name, const char* key=KEY, const char* tmpl=TMPL );

  static bool HasExt(const char* path, const char* ext);
  static std::string SwapExt(const char* path, const char* aext, const char* bext);


};


#endif
