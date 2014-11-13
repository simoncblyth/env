#ifndef G4DAECHROMAPHOTONLIST_H
#define G4DAECHROMAPHOTONLIST_H

//#include "G4DAEChroma/G4DAESerializable.hh"
#include "G4DAEChroma/G4DAEPhotons.hh"
#include <string>

class ChromaPhotonList ; 
class G4DAEBuffer ; 

/*
  Specialization of ancient ChromaPhotonList to 
  provide G4DAESerializable capabilities
  
*/

class G4DAEChromaPhotonList : public G4DAEPhotons {

  static const char* TMPL ; 
  static const char* KEY ; 
  static const char* SHAPE ;  /*not used, here to match NPL*/

public:
  G4DAEChromaPhotonList(ChromaPhotonList* cpl);
  G4DAEChromaPhotonList(G4DAEPhotons* src);
  G4DAEChromaPhotonList(std::size_t itemcapacity);
  virtual ~G4DAEChromaPhotonList();
  static G4DAEChromaPhotonList* Load(const char* evt, const char* key=KEY, const char* tmpl=TMPL );
  static G4DAEChromaPhotonList* LoadPath(const char* path, const char* key=KEY);

public:
  // G4DAEPhotons
  void AddPhoton(G4ThreeVector pos, G4ThreeVector mom, G4ThreeVector pol, float _t, float _wavelength, int _pmtid=-1);
  void GetPhoton(size_t index, G4ThreeVector& pos, G4ThreeVector& mom, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const ; 
  void Print(const char* msg="G4DAEChromaPhotonList::Print") const ;
  void Details(bool hit) const ;
  std::size_t GetCount() const ;
  std::string GetDigest() const ;
  void ClearAll();

public:
   void Save(const char* evt, const char* key=KEY, const char* tmpl=TMPL );
   void SavePath(const char* path, const char* key=KEY);

public:
  // fulfil Serializable protocol 
  G4DAEChromaPhotonList* CreateOther(char* bytes, size_t size); 
  // "CreateOther" would be more natural as a "Create"  classmethod, but that is not convenient for "protocols" 
  virtual void SaveToBuffer();
  virtual const char* GetBufferBytes();
  virtual std::size_t GetBufferSize();
  virtual void DumpBuffer();

private:
  G4DAEBuffer* m_buffer ;
  ChromaPhotonList* m_cpl ;  

};



#endif
