#ifndef _G4DAEUTIL_INCLUDED_
#define _G4DAEUTIL_INCLUDED_

#include <string>

class G4DAEUtil 
{
public:

   static void replaceAll(std::string& id, std::string const& from, std::string const& to);
   static int transformNCName( std::string& id, bool encode );
   static int decodeNCName( std::string& id );
   static int encodeNCName( std::string& id );
   static int testNCName( std::string& id );
   static int testNCNameDemo();

};

#endif

