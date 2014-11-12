#ifndef G4DAEPHOTONLIST_H
#define G4DAEPHOTONLIST_H

#include <vector>
#include <string>

#include <G4ThreeVector.hh>

#include "G4DAEChroma/G4DAESerializable.hh"
#include "G4DAEChroma/G4DAEPhotons.hh"

class G4DAEArray ;

class G4DAEPhotonList : public G4DAEPhotons  {

public:
  G4DAEPhotonList( G4DAEPhotons* arr );
  G4DAEPhotonList( G4DAEArray* arr );
  G4DAEPhotonList( std::size_t itemcapacity = 0, float* data = NULL);
  virtual ~G4DAEPhotonList();

public:
  // G4DAEPhotons protocol
  void AddPhoton(G4ThreeVector pos, G4ThreeVector mom, G4ThreeVector pol, float _t, float _wavelength, int _pmtid=-1); 
  void GetPhoton(size_t index, G4ThreeVector& pos, G4ThreeVector& mom, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const;
  void Print(const char* msg="G4DAEPhotonList::Print") const ; 
  void Details(bool hit) const ;
  std::size_t GetPhotonCount() const ;
  std::string GetPhotonDigest() const ;
  void ClearAllPhotons();

public:
  // G4DAESerializable
  G4DAEPhotonList* CreateOther(char* buffer, std::size_t buflen);
  void SaveToBuffer();
  void DumpBuffer();
  const char* GetBufferBytes();
  std::size_t GetBufferSize();

public:
  // other  
  static std::string GetPath( const char* evt="dummy" , const char* tmpl="DAE_PATH_TEMPLATE_NPY");   
  static G4DAEPhotonList* Load(const char* evt="1", const char* key="GPL", const char* tmpl="DAE_PATH_TEMPLATE_NPY" );
  void Save(const char* evt="dummy", const char* key="GPL", const char* tmpl="DAE_PATH_TEMPLATE_NPY" );

  static G4DAEPhotonList* LoadPath(const char* path, const char* key="GPL");
  void SavePath(const char* path, const char* key="GPL");

private:
   G4DAEArray* m_array ;



};

#endif

