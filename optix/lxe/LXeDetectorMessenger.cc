#include "LXeDetectorMessenger.hh"
#include "LXeDetectorConstruction.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcommand.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4Scintillation.hh"


LXeDetectorMessenger::LXeDetectorMessenger(LXeDetectorConstruction* detector)
 : fLXeDetector(detector)
{
  //Setup a command directory for detector controls with guidance
  fDetectorDir = new G4UIdirectory("/LXe/detector/");
  fDetectorDir->SetGuidance("Detector geometry control");

  fVolumesDir = new G4UIdirectory("/LXe/detector/volumes/");
  fVolumesDir->SetGuidance("Enable/disable volumes");
 
  //Various commands for modifying detector geometry
  fDimensionsCmd =
    new G4UIcmdWith3VectorAndUnit("/LXe/detector/dimensions",this);
  fDimensionsCmd->SetGuidance("Set the dimensions of the detector volume.");
  fDimensionsCmd->SetParameterName("scint_x","scint_y","scint_z",false);
  fDimensionsCmd->SetDefaultUnit("cm");
  fDimensionsCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fDimensionsCmd->SetToBeBroadcasted(false);

  fHousingThicknessCmd = new G4UIcmdWithADoubleAndUnit
    ("/LXe/detector/housingThickness",this);
  fHousingThicknessCmd->SetGuidance("Set the thickness of the housing.");
  fHousingThicknessCmd->SetParameterName("d_mtl",false);
  fHousingThicknessCmd->SetDefaultUnit("cm");
  fHousingThicknessCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fHousingThicknessCmd->SetToBeBroadcasted(false);

  fPmtRadiusCmd = new G4UIcmdWithADoubleAndUnit
    ("/LXe/detector/pmtRadius",this);
  fPmtRadiusCmd->SetGuidance("Set the radius of the PMTs.");
  fPmtRadiusCmd->SetParameterName("radius",false);
  fPmtRadiusCmd->SetDefaultUnit("cm");
  fPmtRadiusCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fPmtRadiusCmd->SetToBeBroadcasted(false);

  fNxCmd = new G4UIcmdWithAnInteger("/LXe/detector/nx",this);
  fNxCmd->SetGuidance("Set the number of PMTs along the x-dimension.");
  fNxCmd->SetParameterName("nx",false);
  fNxCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fNxCmd->SetToBeBroadcasted(false);

  fNyCmd = new G4UIcmdWithAnInteger("/LXe/detector/ny",this);
  fNyCmd->SetGuidance("Set the number of PMTs along the y-dimension.");
  fNyCmd->SetParameterName("ny",false);
  fNyCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fNyCmd->SetToBeBroadcasted(false);

  fNzCmd = new G4UIcmdWithAnInteger("/LXe/detector/nz",this);
  fNzCmd->SetGuidance("Set the number of PMTs along the z-dimension.");
  fNzCmd->SetParameterName("nz",false);
  fNzCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fNzCmd->SetToBeBroadcasted(false);

  fSphereCmd = new G4UIcmdWithABool("/LXe/detector/volumes/sphere",this);
  fSphereCmd->SetGuidance("Enable/Disable the sphere.");
  fSphereCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fSphereCmd->SetToBeBroadcasted(false);

  fReflectivityCmd = new G4UIcmdWithADouble("/LXe/detector/reflectivity",this);
  fReflectivityCmd->SetGuidance("Set the reflectivity of the housing.");
  fReflectivityCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fReflectivityCmd->SetToBeBroadcasted(false);

  fWlsCmd = new G4UIcmdWithABool("/LXe/detector/volumes/wls",this);
  fWlsCmd->SetGuidance("Enable/Disable the WLS slab");
  fWlsCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fWlsCmd->SetToBeBroadcasted(false);

  fLxeCmd = new G4UIcmdWithABool("/LXe/detector/volumes/lxe",this);
  fLxeCmd->SetGuidance("Enable/Disable the main detector volume.");
  fLxeCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fLxeCmd->SetToBeBroadcasted(false);

  fNFibersCmd = new G4UIcmdWithAnInteger("/LXe/detector/nfibers",this);
  fNFibersCmd->SetGuidance("Set the number of WLS fibers in the WLS slab.");
  fNFibersCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fNFibersCmd->SetToBeBroadcasted(false);

  fDefaultsCmd = new G4UIcommand("/LXe/detector/defaults",this);
  fDefaultsCmd->SetGuidance("Set all detector geometry values to defaults.");
  fDefaultsCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  fDefaultsCmd->SetToBeBroadcasted(false);

  fMainScintYield=new G4UIcmdWithADouble("/LXe/detector/MainScintYield",this);
  fMainScintYield->SetGuidance("Set scinitillation yield of main volume.");
  fMainScintYield->SetGuidance("Specified in photons/MeV");
  fMainScintYield->AvailableForStates(G4State_PreInit,G4State_Idle);
  fMainScintYield->SetToBeBroadcasted(false);

  fWLSScintYield = new G4UIcmdWithADouble("/LXe/detector/WLSScintYield",this);
  fWLSScintYield->SetGuidance("Set scintillation yield of WLS Slab");
  fWLSScintYield->SetGuidance("Specified in photons/MeV");
  fWLSScintYield->AvailableForStates(G4State_PreInit,G4State_Idle);
  fWLSScintYield->SetToBeBroadcasted(false);
}


LXeDetectorMessenger::~LXeDetectorMessenger()
{
  delete fDimensionsCmd;
  delete fHousingThicknessCmd;
  delete fPmtRadiusCmd;
  delete fNxCmd;
  delete fNyCmd;
  delete fNzCmd;
  delete fDetectorDir;
  delete fVolumesDir;
  delete fDefaultsCmd;
  delete fSphereCmd;
  delete fWlsCmd;
  delete fLxeCmd;
  delete fNFibersCmd;
  delete fReflectivityCmd;
  delete fMainScintYield;
  delete fWLSScintYield;
}


void LXeDetectorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if( command == fDimensionsCmd ){
    fLXeDetector->SetDimensions(fDimensionsCmd->GetNew3VectorValue(newValue));
  }
  else if (command == fHousingThicknessCmd){
    fLXeDetector->SetHousingThickness(fHousingThicknessCmd
                                     ->GetNewDoubleValue(newValue));
  }
  else if (command == fPmtRadiusCmd){
    fLXeDetector->SetPMTRadius(fPmtRadiusCmd->GetNewDoubleValue(newValue));
  }
  else if (command == fNxCmd){
    fLXeDetector->SetNX(fNxCmd->GetNewIntValue(newValue));
  }
  else if (command == fNyCmd){
    fLXeDetector->SetNY(fNyCmd->GetNewIntValue(newValue));
  }
  else if (command == fNzCmd){
    fLXeDetector->SetNZ(fNzCmd->GetNewIntValue(newValue));
  }
  else if (command == fDefaultsCmd){
    fLXeDetector->SetDefaults();
  }
  else if (command == fSphereCmd){
    fLXeDetector->SetSphereOn(fSphereCmd->GetNewBoolValue(newValue));
  }
  else if (command == fReflectivityCmd){
    fLXeDetector
      ->SetHousingReflectivity(fReflectivityCmd->GetNewDoubleValue(newValue));
  }
  else if (command == fWlsCmd){
    fLXeDetector->SetWLSSlabOn(fWlsCmd->GetNewBoolValue(newValue));
  }
  else if (command == fLxeCmd){
    fLXeDetector->SetMainVolumeOn(fLxeCmd->GetNewBoolValue(newValue));
  }
  else if (command == fNFibersCmd){
    fLXeDetector->SetNFibers(fNFibersCmd->GetNewIntValue(newValue));
  }
  else if (command == fMainScintYield){
   fLXeDetector->SetMainScintYield(fMainScintYield->GetNewDoubleValue(newValue));
  }
  else if (command == fWLSScintYield){
    fLXeDetector->SetWLSScintYield(fWLSScintYield->GetNewDoubleValue(newValue));
  }
}
