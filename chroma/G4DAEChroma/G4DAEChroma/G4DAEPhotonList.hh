#ifndef G4DAEPHOTONLIST_H
#define G4DAEPHOTONLIST_H

#include <vector>
#include <string>

#include <G4ThreeVector.hh>

//#include "G4DAEChroma/G4DAESerializable.hh"
#include "G4DAEChroma/G4DAEPhotons.hh"

class G4DAEArray ;
class G4DAEMetadata ; 
class G4DAETransformCache ;

class G4DAEPhotonList : public G4DAEPhotons  {

  static const char* TMPL ;
  static const char* SHAPE ;
  static const char* KEY ;

public:
  G4DAEPhotonList* Slice( int a, int b );
  G4DAEPhotonList( G4DAEPhotons* arr, int a=0, int b=0 );
  G4DAEPhotonList( G4DAEArray* arr );
  G4DAEPhotonList( std::size_t itemcapacity = 0, float* data = NULL);
  virtual ~G4DAEPhotonList();

public:
  // G4DAEPhotons protocol
  void AddPhoton(G4ThreeVector pos, G4ThreeVector mom, G4ThreeVector pol, float _t, float _wavelength, int _pmtid=-1); 
  void GetPhoton(size_t index, G4ThreeVector& pos, G4ThreeVector& mom, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const;
  void Print(const char* msg="G4DAEPhotonList::Print") const ; 
  void Details(bool hit) const ;


  std::size_t GetCount() const ;
  std::string GetDigest() const ;
  void ClearAll();

public:
  // G4DAESerializable
  G4DAEPhotonList* CreateOther(char* buffer, std::size_t buflen);
  void SaveToBuffer();
  void DumpBuffer();
  const char* GetBufferBytes();
  std::size_t GetBufferSize();
  const char* GetMagic();

  G4DAEMetadata* GetLink();
  void SetLink(G4DAEMetadata* link);
  //G4DAEMetadata* GetLastLink();
  //void AddLink(G4DAEMetadata* link );



public:
  // G4DAEArray persistency 
  static std::string GetPath( const char* evt, const char* tmpl=TMPL);   
  static G4DAEPhotonList* Load(const char* evt, const char* key=KEY, const char* tmpl=TMPL );
  static G4DAEPhotonList* LoadPath(const char* path, const char* key=KEY);

  void Save(const char* evt, const char* key=KEY, const char* tmpl=TMPL );
  void SavePath(const char* path, const char* key=KEY);

private:
   G4DAEArray* m_array ;
   G4DAEMetadata* m_link ; 


};

#endif

