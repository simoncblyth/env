#include "GaudiKernel/DeclareFactoryEntries.h"
#include "GaudiKernel/PropertyMgr.h"

#include "G4VPhysicalVolume.hh"
#include "G4TransportationManager.hh"

#ifdef EXPORT_G4GDML
#include "G4GDMLParser.hh"
#endif

#ifdef EXPORT_G4DAE
#include "G4DAEParser.hh"
#endif

#ifdef EXPORT_G4WRL
#include "G4UImanager.hh"
#endif


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

/** performe the action at the begin of each run 
 *  @param run pointer to Geant4 run object 
 */
void GiGaRunActionGDML::BeginOfRunAction( const G4Run* run )
{

  if( 0 == run ) 
    { Warning("BeginOfRunAction:: G4Run* points to NULL!") ; }

   G4VPhysicalVolume* wpv = G4TransportationManager::GetTransportationManager()->
      GetNavigatorForTracking()->GetWorldVolume();



#ifdef EXPORT_G4WRL
   G4UImanager* ui = G4UImanager::GetUIpointer() ; 
   ui->ApplyCommand("/vis/open VRML2FILE");
   ui->ApplyCommand("/vis/viewer/set/culling global false");
   ui->ApplyCommand("/vis/viewer/set/culling coveredDaughters false");
   ui->ApplyCommand("/vis/drawVolume");
   ui->ApplyCommand("/vis/viewer/flush");
#endif



#ifdef EXPORT_G4GDML
   G4String gdmlFilePath("g4_00.gdml");
   G4GDMLParser gdmlparser ;
   if(wpv)
   {
       std::cout << "GiGaRunActionGDML::BeginOfRunAction writing GDML to " << gdmlFilePath << std::endl ;
       gdmlparser.Write(gdmlFilePath, wpv);
   } 
   else
   {
       std::cout << "GiGaRunActionGDML::BeginOfRunAction  Null pointer to world pv" << std::endl;
   }
#endif

#ifdef EXPORT_G4DAE
   G4String daeFilePath("g4_00.dae");
   G4DAEParser daeparser ;
   if(wpv)
   {
       std::cout << "GiGaRunActionGDML::BeginOfRunAction writing COLLADA to " << daeFilePath << std::endl ;
       daeparser.Write(daeFilePath, wpv);
   } 
   else
   {
       std::cout << "GiGaRunActionGDML::BeginOfRunAction  Null pointer to world pv" << std::endl;
   }
#endif




};

/** performe the action at the end of each run 
 *  @param run pointer to Geant4 run object 
 */
// ============================================================================
void GiGaRunActionGDML::EndOfRunAction( const G4Run* run )
{
  if( 0 == run ) 
    { Warning("EndOfRunAction:: G4Run* points to NULL!") ; }
};


