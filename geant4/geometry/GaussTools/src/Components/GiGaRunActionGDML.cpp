#include "GaudiKernel/DeclareFactoryEntries.h"
#include "GaudiKernel/PropertyMgr.h"

#include "G4VPhysicalVolume.hh"
#include "G4TransportationManager.hh"
#include "G4SolidStore.hh"

#ifdef EXPORT_G4GDML
#include "G4GDMLParser.hh"
#endif

#ifdef EXPORT_G4DAE
#include "G4DAEParser.hh"
#endif

#ifdef EXPORT_G4WRL
#include "G4UImanager.hh"
#endif


#include <sstream>
#include <fstream>
#include <stdlib.h>  

/// Local 
#include "GiGaRunActionGDML.h"

// ============================================================================
/** @file 
 *
 *  Implementation file for class : GiGaRunActionGDML
 *
 */
// ============================================================================

// Declaration of the Tool Factory
DECLARE_TOOL_FACTORY( GiGaRunActionGDML );

/** standard constructor 
 *  @see GiGaPhysListBase
 *  @see GiGaBase 
 *  @see AlgTool 
 *  @param type type of the object (?)
 *  @param name name of the object
 *  @param parent  pointer to parent object
 *
 *  Implementation based on 
 *     external/build/LCG/geant4.9.2.p01/examples/extended/persistency/gdml/G02/src/DetectorConstruction.cc 
 */
GiGaRunActionGDML::GiGaRunActionGDML
( const std::string& type   ,
  const std::string& name   ,
  const IInterface*  parent ) 
  : GiGaRunActionBase( type , name , parent )
{  
};

GiGaRunActionGDML::~GiGaRunActionGDML()
{
};

void GiGaRunActionGDML::WriteGDML(G4VPhysicalVolume* wpv, const G4String& path )
{
#ifdef EXPORT_G4GDML
   if(path.length() == 0 || wpv == 0){
       std::cout << "GiGaRunActionGDML::WriteGDML invalid path OR NULL PV  " << path << std::endl ;
       return ;  
   }
   std::cout << "GiGaRunActionGDML::WriteGDML to " << path << std::endl ;
   G4GDMLParser parser ;
   parser.Write(path, wpv);
#endif
}

void GiGaRunActionGDML::WriteDAE(G4VPhysicalVolume* wpv, const G4String& path, G4bool recreatePoly  )
{
#ifdef EXPORT_G4DAE
   if(path.length() == 0 || wpv == 0){
       std::cout << "GiGaRunActionGDML::WriteDAE invalid path OR NULL PV  " << path << std::endl ;
       return ;  
   }
   std::cout << "GiGaRunActionGDML::WriteDAE to " << path << " recreatePoly " << recreatePoly << std::endl ;
   G4DAEParser parser ;
   G4bool refs = true ; 
   parser.Write(path, wpv, refs, recreatePoly );
#endif
}

G4bool GiGaRunActionGDML::FileExists(const char *fileName){
    std::ifstream infile(fileName);
    return infile.good();
}

G4String GiGaRunActionGDML::FilePath( const G4String& base , G4int index, const G4String& ext , G4bool wantfree )
{
   /*
         base 
              prefix path eg "/path/to/directory/g4_" "g4_" 
         counter
              0,1,..,99 
         ext 
              ".gdml"
         wantfree
              if true return a blank string if the path is already existing
              if false return path regardless of existance 
             
   */
   std::ostringstream ss;
   ss << base << std::setw(2) << std::setfill('0') << index << ext << std::setfill(' ') ;
   std::string path = ss.str();
   G4bool exists_ = FileExists(path.c_str());
   if( wantfree && exists_ ){
        //std::cout << "path " << path << " exists already " << std::endl ;
        return G4String("") ; 
   } 
   return G4String(path);
}

G4String GiGaRunActionGDML::FreeFilePath( const G4String& base, const G4String& ext )
{
    G4int imax(99);
    G4int i(0);
    G4String path("") ; 
    while(path.length() == 0){
        path = FilePath( base,  i++, ext , true );
        if( i == imax ) break ; 
    }
    std::cout << "FreeFilePath  return " << path << " i " << i <<  std::endl ;
    return path ; 
}

void GiGaRunActionGDML::CleanSolidStore()
{
    std::cout << "GiGaRunActionGDML::CleanSolidStore deleting all solids from the store " <<  std::endl ;
    G4SolidStore::Clean();
}


void GiGaRunActionGDML::InitVis(const char* driver)
{
#ifdef EXPORT_G4WRL
   G4UImanager* ui = G4UImanager::GetUIpointer() ; 

   G4String vis_open("/vis/open ");
   vis_open += driver ; 
   ui->ApplyCommand(vis_open.c_str());
   ui->ApplyCommand("/vis/geometry/list all");
   ui->ApplyCommand("/vis/viewer/set/culling global false");
   ui->ApplyCommand("/vis/viewer/set/culling coveredDaughters false");
   //ui->ApplyCommand("/vis/viewer/set/lineSegmentsPerCircle 100");    
#endif
}

void GiGaRunActionGDML::FlushVis(const char* driver)
{
#ifdef EXPORT_G4WRL
   G4UImanager* ui = G4UImanager::GetUIpointer() ; 
   ui->ApplyCommand("/vis/drawVolume");
   ui->ApplyCommand("/vis/viewer/flush");
#endif
}


void GiGaRunActionGDML::WriteVis(const char* driver)
{
#ifdef EXPORT_G4WRL
   G4UImanager* ui = G4UImanager::GetUIpointer() ; 
   G4String drv(driver); 
   G4String vis_open("/vis/open ");
   vis_open += drv ; 
   G4cout << "GiGaRunActionGDML::WriteVis " << vis_open << G4endl ; 
   ui->ApplyCommand(vis_open);
   G4cout << "GiGaRunActionGDML::WriteVis list geom " << G4endl ; 
   ui->ApplyCommand("/vis/geometry/list all");
   G4cout << "GiGaRunActionGDML::WriteVis set culling 1  " << G4endl ; 
   ui->ApplyCommand("/vis/viewer/set/culling global false");
   G4cout << "GiGaRunActionGDML::WriteVis set culling 2  " << G4endl ; 
   ui->ApplyCommand("/vis/viewer/set/culling coveredDaughters false");
   //ui->ApplyCommand("/vis/viewer/set/lineSegmentsPerCircle 100");    
   G4cout << "GiGaRunActionGDML::WriteVis drawVolume  " << G4endl ; 
   ui->ApplyCommand("/vis/drawVolume");
   G4cout << "GiGaRunActionGDML::WriteVis flush  " << G4endl ; 
   ui->ApplyCommand("/vis/viewer/flush");
   G4cout << "GiGaRunActionGDML::WriteVis done  " << G4endl ; 
#endif
}

G4String GiGaRunActionGDML::GetEnv( const char* envvar , const char* def )
{
   char const* tmp = getenv(envvar);   // no trailing slash 
   G4String val = ( tmp == NULL ) ? def : tmp ;  
   return val ; 
}

void GiGaRunActionGDML::AbruptExit()
{
   std::cout << "GiGaRunActionGDML::AbruptExit due to G4DAE_EXPORT_EXIT: " << std::endl ;  
   exit(0);
}


void GiGaRunActionGDML::BeginOfRunAction( const G4Run* run )
{

  if( 0 == run ) 
    { Warning("BeginOfRunAction:: G4Run* points to NULL!") ; }

   G4VPhysicalVolume* wpv = G4TransportationManager::GetTransportationManager()->
      GetNavigatorForTracking()->GetWorldVolume();


   // setup output directories
   G4String xdir = GetEnv("G4DAE_EXPORT_DIR", ".");   // no trailing slash 
   G4String vdir(xdir);
   vdir += "/" ;       // VRML needs a trailing slash 
   setenv("G4VRMLFILE_DEST_DIR", vdir.c_str(), 0 );

   G4String base(xdir);
   base += "/g4_" ; 

   // write geometry, multiple times and interleaved for DAE/WRL interference testing 
   G4String xseq = GetEnv("G4DAE_EXPORT_SEQUENCE","VGD");
   const char* seq = xseq.c_str();

   for (int i = 0; i < strlen(seq); i++){
       char c = seq[i];
       std::cout << "GiGaRunActionGDML::BeginOfRunAction i " << i << " c " << c << std::endl ;  
       switch (c) 
       {
          case 'V':
                 WriteVis("VRML2FILE");
                 break;
          case 'I':
                 InitVis("VRML2FILE");
                 break;
          case 'F':
                 FlushVis("VRML2FILE");
                 break;
          case 'G':
                 WriteGDML( wpv, FreeFilePath(base, ".gdml"));
                 break;
          case 'A':
                 WriteDAE( wpv, FreeFilePath(base, ".dae"), true );
                 break;
          case 'D':
                 WriteDAE( wpv, FreeFilePath(base, ".dae"), false );
                 break;
          case 'C':
                 CleanSolidStore();
                 break;
          case 'X':
                 AbruptExit();
                 break;
       }
   }

  
};

void GiGaRunActionGDML::EndOfRunAction( const G4Run* run )
{
  if( 0 == run ) 
    { Warning("EndOfRunAction:: G4Run* points to NULL!") ; }
};


