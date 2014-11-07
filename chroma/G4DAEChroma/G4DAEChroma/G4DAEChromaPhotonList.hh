#ifndef G4DAECHROMAPHOTONLIST_H
#define G4DAECHROMAPHOTONLIST_H

#include "G4DAEChroma/G4DAESerializable.hh"
#include "Chroma/ChromaPhotonList.hh"

class G4DAEBuffer ; 

class G4DAEChromaPhotonList : public G4DAESerializable,  public ChromaPhotonList {

  G4DAEChromaPhotonList();
  virtual ~G4DAEChromaPhotonList();
  static G4DAEChromaPhotonList* Create(const char* bytes, size_t size);

public:
  // fulfil Serializable protocol 
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
