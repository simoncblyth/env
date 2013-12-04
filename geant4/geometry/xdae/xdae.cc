
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VUserDetectorConstruction.hh"

#include "G4GDMLParser.hh"
#include "G4DAEParser.hh"






class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction() : fWorld(0) {};
   ~DetectorConstruction(){};
    G4VPhysicalVolume* Construct(){
        return fWorld ;
}
    G4VPhysicalVolume* GetWorldVolume(){ return fWorld ; }


    void ApplyCommand(const char* cmd){
       G4UImanager* ui = G4UImanager::GetUIpointer() ; 
       G4cout << "ApplyCommand " << cmd << G4endl ; 
       ui->ApplyCommand(cmd);
    }

    G4String GetEnv( const char* envvar , const char* def )
    {
        char const* tmp = getenv(envvar);   // no trailing slash 
        G4String val = ( tmp == NULL ) ? def : tmp ;  
        return val ; 
    }

   void PrepVRML(const G4String& xdir ){
       G4String vdir(xdir);
       vdir += "/" ;       // VRML needs a trailing slash 
       setenv("G4VRMLFILE_DEST_DIR", vdir.c_str(), 0 );
    }

    void WriteVRML(){
       ApplyCommand("/vis/open VRML2FILE");
       ApplyCommand("/vis/geometry/list all");
       ApplyCommand("/vis/viewer/set/culling global false");
       ApplyCommand("/vis/viewer/set/culling coveredDaughters false");
       ApplyCommand("/vis/drawVolume");
       ApplyCommand("/vis/viewer/flush");
    }

    void ReadGDML(const G4String& filename, G4bool validate=false){
        fGDMLParser.Read(filename, validate);
        fWorld = fGDMLParser.GetWorldVolume();
    }

    void WriteGDML(const G4String& filename){
        fGDMLParser.Write(filename, fWorld);
    }

    void ReadDAE(const G4String& filename, G4bool validate=false){
        fDAEParser.Read(filename, validate);
        fWorld = fDAEParser.GetWorldVolume();
    }

    void WriteDAE(const G4String& filename){
        fDAEParser.Write(filename, fWorld);
    }


  private:
    G4DAEParser  fDAEParser;
    G4GDMLParser fGDMLParser;
    G4VPhysicalVolume* fWorld ;

};




int main(int argc, char** argv)
{

   G4RunManager* runManager = new G4RunManager;
   G4VisManager* visManager = new G4VisExecutive;
   visManager->Initialize();

   DetectorConstruction* dc = new DetectorConstruction;
   dc->ReadGDML("/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml");

   runManager->SetUserInitialization(dc);
   runManager->Initialize();

   G4String xdir = dc->GetEnv("G4DAE_EXPORT_DIR", ".");   // no trailing slash 
   dc->PrepVRML(xdir);
   G4String base(xdir);
   base += "/g4_" ; 
   dc->WriteDAE(base+"00.dae");
   dc->WriteVRML();

}

