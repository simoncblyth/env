#ifndef _G4DAEWRITESETUP_INCLUDED_
#define _G4DAEWRITESETUP_INCLUDED_

#include "G4DAEWriteSolids.hh"

class G4DAEWriteSetup : public G4DAEWriteSolids
{
 private:

   void SetupWrite(xercesc::DOMElement*,  const G4LogicalVolume* );


};

#endif
