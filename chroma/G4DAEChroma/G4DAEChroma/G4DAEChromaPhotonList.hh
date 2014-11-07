#ifndef G4DAECHROMAPHOTONLIST_H
#define G4DAECHROMAPHOTONLIST_H

#include "G4DAEChroma/G4DAESerializable.hh"
#include "Chroma/ChromaPhotonList.hh"

class G4DAEBuffer ; 

/*
  Specialization of ancient ChromaPhotonList to 
  provide G4DAESerializable capabilities
  
*/

class G4DAEChromaPhotonList : public G4DAESerializable,  public ChromaPhotonList {

public:
  G4DAEChromaPhotonList(std::size_t itemcapacity);
  virtual ~G4DAEChromaPhotonList();
  static G4DAEChromaPhotonList* Load(const char* evt="1", const char* key="CPL", const char* tmpl="DAE_PATH_TEMPLATE" );

public:
  // fulfil Serializable protocol 
  G4DAEChromaPhotonList* Create(char* bytes, size_t size);
  virtual void SaveToBuffer();
  virtual const char* GetBufferBytes();
  virtual std::size_t GetBufferSize();
  virtual void DumpBuffer();

private:
  G4DAEBuffer* m_buffer ;
  
   // hmm do i need the below if cast down to CPL ?
   //! exclamation mark is transient signal to ROOT TObject serialization

};



#endif
