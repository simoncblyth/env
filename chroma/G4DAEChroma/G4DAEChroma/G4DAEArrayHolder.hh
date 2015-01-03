#ifndef G4DAEARRAYHOLDER_H
#define G4DAEARRAYHOLDER_H

#include <string>

class G4DAEArray ;
class G4DAEMetadata ;

#include "G4DAEChroma/G4DAESerializable.hh"
#include "G4DAEChroma/G4DAEMap.hh"

class G4DAEArrayHolder : public G4DAESerializable {

public:
  G4DAEArrayHolder( G4DAEArrayHolder* other ); // CAUTION: copy ctor that just steals pointers without copying 
  G4DAEArrayHolder( G4DAEArrayHolder* other, int start, int stop, int step ); 
  G4DAEArrayHolder( G4DAEArray* array );
  G4DAEArrayHolder( G4DAEArray* array, int start, int stop, int step );
  G4DAEArrayHolder( std::size_t itemcapacity = 0, float* data = NULL, const char* shape = "3,3" );
  virtual ~G4DAEArrayHolder();

  void SetArray(G4DAEArray* array);
  G4DAEArray* GetArray(); 

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
  void CreateLink();
  void AddMap(const char* name, Map_t& meta);
  void SetKV(const char* name, const char* key, const char* val);
  void SetKV(const char* name, const char* key, int ival);


protected:
   G4DAEArray* m_array ;
   G4DAEMetadata* m_link ; 


};

#endif





