#ifndef G4DAEARRAYHOLDER_H
#define G4DAEARRAYHOLDER_H

#include <string>

class G4DAEArray ;

class G4DAEArrayHolder {

public:
  G4DAEArrayHolder( G4DAEArray* array );
  G4DAEArrayHolder( std::size_t itemcapacity = 0, float* data = NULL, const char* shape = "3,3" );
  virtual ~G4DAEArrayHolder();

public:
  virtual void Print(const char* msg="G4DAEArrayHolder::Print") const ; 
  virtual std::string GetDigest() const ;  
  virtual void ClearAll();
  virtual std::size_t GetCount() const ;

  /*
  virtual void Save(const char* evt, const char* key, const char* tmpl );
  virtual void SavePath(const char* path, const char* key);
  */

  virtual float* GetNextPointer();

protected:
   G4DAEArray* m_array ;

};

#endif





