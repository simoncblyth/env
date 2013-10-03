#ifndef G4DAEFILE_SCENE_HANDLER_HH
#define G4DAEFILE_SCENE_HANDLER_HH

#include <fstream>

#include "globals.hh"
#include "G4LogicalVolume.hh"
#include "G4VSceneHandler.hh"


class G4DAEFile;
class G4VisAttributes;

class G4DAEFileSceneHandler: public G4VSceneHandler {

  friend class G4DAEFileViewer;

// methods (public) 
public:
	G4DAEFileSceneHandler(G4DAEFile& system, const G4String& name = "");
	virtual ~G4DAEFileSceneHandler();
	void AddSolid(const G4Box&);
	void AddSolid(const G4Cons&);
	void AddSolid(const G4Tubs&);
	void AddSolid(const G4Trd&);
	void AddSolid(const G4Trap&);
	void AddSolid(const G4Sphere&);
        void AddSolid(const G4Para&);
	void AddSolid(const G4Torus&);
        void AddSolid ( const G4Polycone& polycone ) {
          G4VSceneHandler::AddSolid (polycone);
        }
        void AddSolid ( const G4Polyhedra& polyhedra) {
          G4VSceneHandler::AddSolid (polyhedra);
        }
        void AddSolid(const G4VSolid&);
        void AddCompound ( const G4VTrajectory& traj) {
          G4VSceneHandler::AddCompound(traj);
        }
        void AddCompound ( const G4VHit& hit) {
          G4VSceneHandler::AddCompound(hit);
        }

	void BeginPrimitives(const G4Transform3D& objectTransformation);
	void EndPrimitives();

	void AddPrimitive(const G4Polyline&);
	void AddPrimitive(const G4Polyhedron&);
    void AddPrimitive(const G4NURBS&);
	void AddPrimitive(const G4Text&); 
	void AddPrimitive(const G4Circle&);
	void AddPrimitive(const G4Square&);
	void AddPrimitive (const G4Polymarker& polymarker)
		{ G4VSceneHandler::AddPrimitive (polymarker); }
        void AddPrimitive (const G4Scale& scale) 
                { G4VSceneHandler::AddPrimitive (scale); }

	void ClearTransientStore();  // Used for triggering detector re-drawing.

	void BeginModeling();
	void EndModeling();

	void VRMLBeginModeling();
	void VRMLEndModeling();

	void connectPort();
	void closePort();

// methods (private) 
private:

	void      SendMaterialNode          ( const G4VisAttributes*  pAV ); 
	void      SendMaterialNode          ();

	void      SendLineColor             ( const G4VisAttributes*  pAV ); 
	void      SendMarkerColor           ( const G4VMarker&  mark ) ;
	void      SendMarkerWorldPosition   ( const G4VMarker&  mark ) ;

	G4double  GetMarkerHalfSize         ( const G4VMarker&  mark ) ;
	void      GetMarkerWorldPosition    (	const G4VMarker&  mark , 
						double* pX             ,
						double* pY             ,
						double* pZ              ) ;

	G4bool    isConnected      () { return fFlagDestOpen ; }

	G4bool    IsPVPickable     ()                { return fPVPickable   ;}  
	void      SetPVPickability ( G4bool on_off ) { fPVPickable = on_off ;}  
	G4double  SetPVTransparency ()  ; 
	G4double  GetPVTransparency () { return fPVTransparency ; } 

// data 
private:
	char fVRMLFileDestDir[256] ; 
	char fVRMLFileName[256]    ; 

	G4DAEFile&   fSystem;	// Graphics system for this scene.
	G4bool       fFlagDestOpen ;

	G4int        fMaxFileNum   ;	

	G4bool       fPVPickable ;
	G4double     fPVTransparency ;

	static G4int fSceneIdCount;

public: 
	std::ofstream     fDest ;

};

#endif //G4DAEFILE_SCENE_HANDLER_HH
