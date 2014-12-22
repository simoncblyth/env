#ifndef G4DAEARRAYHOLDER_H
#define G4DAEARRAYHOLDER_H

#include <string>

class G4DAEArray ;
class G4DAEMetadata ;

#include "G4DAEChroma/G4DAESerializable.hh"

class G4DAEArrayHolder : public G4DAESerializable {

public:
  G4DAEArrayHolder( G4DAEArray* array );
  G4DAEArrayHolder( std::size_t itemcapacity = 0, float* data = NULL, const char* shape = "3,3" );
  virtual ~G4DAEArrayHolder();

public:
  virtual void Print(const char* msg="G4DAEArrayHolder::Print") const ; 
  virtual std::string GetDigest() const ;  
  virtual void ClearAll();
  virtual std::size_t GetCount() const ;

  virtual float* GetItemPointer(std::size_t index);
  virtual float* GetNextPointer();

public:
  // G4DAESerializable
  virtual G4DAEArrayHolder* CreateOther(char* bytes, std::size_t size);

  void SaveToBuffer();
  void DumpBuffer();
  const char* GetBufferBytes();
  std::size_t GetBufferSize();
  const char* GetMagic();  

  G4DAEMetadata* GetLink();
  void SetLink(G4DAEMetadata* link);


protected:
   G4DAEArray* m_array ;
   G4DAEMetadata* m_link ; 


};

#endif





