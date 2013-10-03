#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4UIterminal.hh"
#include "G4VisExecutive.hh"

#include "G4VUserDetectorConstruction.hh"
#include "G4GDMLParser.hh"
#include "G4DAEFile.hh"

#include "PhysicsList.hh"


class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction(){};
   ~DetectorConstruction(){};
    G4VPhysicalVolume* Construct(){
        G4String filename = "/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml";
        fParser.Read(filename,false);
        return fParser.GetWorldVolume();
    }
  private:
    G4GDMLParser fParser;

};

   

int main(int argc, char** argv)
{
  G4RunManager* runManager = new G4RunManager;
  DetectorConstruction* detector = new DetectorConstruction;
  runManager->SetUserInitialization(detector);
  runManager->SetUserInitialization(new PhysicsList);
  runManager->Initialize();

  G4VisManager* visManager = new G4VisExecutive;
  visManager->RegisterGraphicsSystem(new G4DAEFile);
  visManager->Initialize();

  G4UImanager* UI = G4UImanager::GetUIpointer();  
  G4UIsession* session = new G4UIterminal();
  UI->ApplyCommand("/control/execute vis.mac");    
  session->SessionStart();
  delete session;

}

