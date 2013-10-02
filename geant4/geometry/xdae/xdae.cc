#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4UIterminal.hh"
#include "G4VisExecutive.hh"

#include "G4VUserDetectorConstruction.hh"
#include "G4GDMLParser.hh"
#include "G4DAEFile.hh"


class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction(){};
   ~DetectorConstruction(){};
    G4VPhysicalVolume* Construct();

  private:
    G4GDMLParser fParser;

};

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    G4String filename = "/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml";
    G4bool validate = false; 
    fParser.Read(filename,validate);
    return parser.GetWorldVolume();
}
   

int main(int argc, char** argv)
{
  G4RunManager* runManager = new G4RunManager;
  DetectorConstruction* detector = new DetectorConstruction;
  runManager->SetUserInitialization(detector);

  G4VisManager* visManager = new G4VisExecutive;
  visManager->RegisterGraphicsSystem(new G4DAEFile);
  visManager->Initialize();

  G4UImanager* UI = G4UImanager::GetUIpointer();  
  G4UIsession* session = new G4UIterminal();
  UI->ApplyCommand("/control/execute vis.mac");    
  session->SessionStart();
  delete session;

}

