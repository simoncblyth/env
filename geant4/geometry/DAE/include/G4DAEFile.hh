#ifndef G4DAEFILE_HH
#define G4DAEFILE_HH

#include "G4VGraphicsSystem.hh"

class G4VSceneHandler;

class G4DAEFile: public G4VGraphicsSystem {
public:
	G4DAEFile(); 
	virtual ~G4DAEFile();
	G4VSceneHandler* CreateSceneHandler(const G4String& name = "");
	G4VViewer*  CreateViewer(G4VSceneHandler&, const G4String& name = "");

};

#endif //G4DAEFILE_HH
