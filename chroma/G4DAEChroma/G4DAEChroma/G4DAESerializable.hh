#ifndef G4DAESERIALIZABLE_H
#define G4DAESERIALIZABLE_H

#include <cstdlib>
class G4DAESerializable ;

class G4DAESerializable {
public:
   //
   // virtual void Populate( const char* bytes, size_t size ) = 0; 
   //     cant work out how to implement this with ROOT TObject deserialization
   //     so fallback to informal "Create" static method
   //
   virtual void SaveToBuffer() = 0 ;
   virtual const char* GetBufferBytes() = 0 ;
   virtual std::size_t GetBufferSize() = 0 ;
   virtual void DumpBuffer() = 0 ;
   virtual G4DAESerializable* Create(char* bytes, std::size_t size) = 0 ;

};

#endif 

