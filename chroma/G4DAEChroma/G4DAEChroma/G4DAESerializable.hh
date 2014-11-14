#ifndef G4DAESERIALIZABLE_H
#define G4DAESERIALIZABLE_H

#include <cstdlib>
class G4DAESerializable ;
class G4DAEMetadata ;

class G4DAESerializable {
public:
   virtual void SaveToBuffer() = 0 ;
   virtual const char* GetBufferBytes() = 0 ;
   virtual std::size_t GetBufferSize() = 0 ;
   virtual void DumpBuffer() = 0 ;
   virtual G4DAESerializable* CreateOther(char* bytes, std::size_t size) = 0 ;

   // impure : do nothing default implementation
   virtual const char* GetMagic();
   virtual G4DAEMetadata* GetLink();
   virtual void SetLink(G4DAEMetadata* link);
   virtual void AddLink(G4DAEMetadata* link);
   virtual G4DAEMetadata* GetLastLink();

};

#endif 

