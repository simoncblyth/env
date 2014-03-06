#ifndef GIGA_GIGARUNACTIONEXPORT_H 
#define GIGA_GIGARUNACTIONEXPORT_H 1

#include "GiGa/GiGaRunActionBase.h"

/** @class GiGaRunActionExport GiGaRunActionExport.h
 *  
 *  A concrete Run Action. 
 *  Exports Geant4 geometry into DAE,GDML,WRL files at the BeginOfRun
 *
 *  Updated from http://svn.cern.ch/guest/lhcb/packages/trunk/Sim/GDMLG4Writer/src/GDMLRunAction.h 
 *
 *  @author Simon Blytyh
 *  @date   
 */

class GiGaRunActionExport: public virtual GiGaRunActionBase
{
  /// friend factory for instantiation
  //friend class GiGaFactory<GiGaRunActionExport>;
  
public:
 
  /** performe the action at the begin of each run 
   *  @param run pointer to Geant4 run object 
   */
  void BeginOfRunAction ( const G4Run* run );
  
  /** performe the action at the end  of each event 
   *  @param run pointer to Geant4 run object 
   */
  void EndOfRunAction   ( const G4Run* run );
  
  //protected:
  
  /** standard constructor 
   *  @see GiGaPhysListBase
   *  @see GiGaBase 
   *  @see AlgTool 
   *  @param type type of the object (?)
   *  @param name name of the object
   *  @param parent  pointer to parent object
   */
  GiGaRunActionExport
  ( const std::string& type   ,
    const std::string& name   ,
    const IInterface*  parent ) ;
  
  // destructor (virtual and protected)
  virtual ~GiGaRunActionExport( );

  void CleanSolidStore();
  void WriteVis(const char* driver);
  void InitVis(const char* driver);
  void FlushVis(const char* driver);
  void AbruptExit();
  void WriteDAE(G4VPhysicalVolume* wpv, const G4String& path, G4bool recreatePoly );
  void WriteGDML(G4VPhysicalVolume* wpv, const G4String& path );

  G4String GetEnv( const char* envvar , const char* def );
  G4String FreeFilePath( const G4String& base, const G4String& ext );
  G4String FilePath( const G4String& base , G4int index, const G4String& ext , G4bool free );
  G4bool FileExists(const char *fileName);
 
private:
  
  ///no default constructor
  GiGaRunActionExport();
  /// no copy constructor 
  GiGaRunActionExport( const GiGaRunActionExport& );  
  /// no assignement 
  GiGaRunActionExport& operator=( const GiGaRunActionExport& );



private:

  std::string m_schemaPath ;
  std::string m_outFilePath ;     

 
};
#endif ///< GIGA_GIGARUNACTIONEXPORT_H
