#include <stdio.h> // sscanf
#include <stdlib.h> // getenv

#include "G4VSceneHandler.hh"

#include "G4DAEFile.hh"
#include "G4DAEFileSceneHandler.hh"
#include "G4DAEFileViewer.hh"



G4DAEFile::G4DAEFile() :
	G4VGraphicsSystem("DAEFILE", "DAEFILE", G4VGraphicsSystem::threeD)
{
}

G4DAEFile::~G4DAEFile()
{
}


G4VSceneHandler* G4DAEFile::CreateSceneHandler(const G4String& name) 
{
	G4VSceneHandler *p = NULL;

	p = new G4DAEFileSceneHandler(*this, name);

	return p;
}

G4VViewer* G4DAEFile::CreateViewer(G4VSceneHandler& scene, const G4String& name)
{
	G4VViewer* pView = NULL;

	G4DAEFileSceneHandler* pScene = (G4DAEFileSceneHandler*)&scene;
	pView = new G4DAEFileViewer(*pScene, name);

	return pView;
}
