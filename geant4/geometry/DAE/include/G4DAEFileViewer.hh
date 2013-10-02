#ifndef G4DAEFILE_VIEWER_HH
#define G4DAEFILE_VIEWER_HH

#include <fstream>
#include "G4VViewer.hh"
#include "globals.hh"

class G4DAEFileSceneHandler;

class G4DAEFileViewer: public G4VViewer {
public:
	G4DAEFileViewer(G4DAEFileSceneHandler& scene, const G4String& name = "");
	virtual ~G4DAEFileViewer();
	void ClearView();
	void DrawView();
	void ShowView();
	void FinishView();
private:
	void SetView(); // Do nothing. SendViewParameters will do its job.
	void SendViewParameters ()  ;

private:
	G4DAEFileSceneHandler& fSceneHandler; // Reference to Graphics Scene for this view.
	std::ofstream&         fDest ;

	G4double      fViewHalfAngle ;	
	G4double      fsin_VHA       ;	

};

#endif //G4DAEFILE_VIEWER_HH
