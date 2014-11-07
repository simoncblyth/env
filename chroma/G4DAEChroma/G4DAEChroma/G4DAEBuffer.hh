#ifndef G4DAEBUFFER_H
#define G4DAEBUFFER_H

#include <cstdlib>

class G4DAEBuffer {

public:
  G4DAEBuffer(std::size_t size, char* bytes=NULL );
  virtual ~G4DAEBuffer();

  virtual std::size_t GetSize() const; 
  virtual char* GetBytes(); 
  virtual void Dump() const ;

private:
    std::size_t      m_size ; 
    char*            m_bytes ;

};


#endif



