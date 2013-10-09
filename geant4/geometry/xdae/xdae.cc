
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
  DetectorConstruction* detector = new DetectorConstruction;
  detector->ReadGDML("/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml");
  detector->WriteDAE("/data1/env/local/env/geant4/geometry/xdae/g4_01.dae");

}

