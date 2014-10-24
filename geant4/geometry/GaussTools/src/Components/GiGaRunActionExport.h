#ifndef GIGA_GIGARUNACTIONEXPORT_H 
#define GIGA_GIGARUNACTIONEXPORT_H 1

#include "GaudiAlg/GaudiTool.h"
#include "GiGa/GiGaRunActionBase.h"
#include "G4Transform3D.hh"
#include "G4NavigationHistory.hh"
#include "G4AffineTransform.hh"

class G4LogicalVolume;
class G4TouchableHistory;  
class IDetectorElement;
class DetectorElement; 

/** @class GiGaRunActionExport GiGaRunActionExport.h
 *  
 *  A concrete Run Action. 
 *  Exports Geant4 geometry into DAE,GDML,WRL files at the BeginOfRun
 *
 *  Updated from http://svn.cern.ch/guest/lhcb/packages/trunk/Sim/GDMLG4Writer/src/GDMLRunAction.h 
 *
 *  @author Simon Blyth
 *  @date   
 */



class GiGaRunActionExport: public virtual GiGaRunActionBase
{
  /// friend factory for instantiation
  //friend class GiGaFactory<GiGaRunActionExport>;
  
public:
 
  typedef std::vector<G4VPhysicalVolume*> PVStack_t;
 

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
  void WriteIdMap(G4VPhysicalVolume* wpv, const G4String& path );
  void WriteDAE(G4VPhysicalVolume* wpv, const G4String& path, G4bool recreatePoly );
  void WriteGDML(G4VPhysicalVolume* wpv, const G4String& path );

  EVolume VolumeType(G4VPhysicalVolume* pv) const;
  void VisitPV( const PVStack_t& pvStack );
  void TraverseVolumeTree(const G4LogicalVolume* const topVol, PVStack_t pvStack );

  G4String GetEnv( const char* envvar , const char* def );
  G4String FreeFilePath( const G4String& base, const G4String& ext );
  G4String FilePath( const G4String& base , G4int index, const G4String& ext , G4bool free );
  G4bool FileExists(const char *fileName);


private:

  // adapted from NuWa-trunk/dybgaudi/Simulation/G4DataHelpers/src/components/TH2DE.{h,cc}
   virtual StatusCode GetBestDetectorElement(const G4TouchableHistory* inHistory,
                                              const IDetectorElement* &outElement,
                                              int& outCompatibility);
 
   typedef std::pair<std::string,std::string> LvPvPair_t;
   typedef std::vector<LvPvPair_t> NameHistory_t;
   typedef std::vector<G4VPhysicalVolume*> TouchableHistory_t;

   const IDetectorElement* FindChildDE(const IDetectorElement* de, NameHistory_t& name_history);
   const IDetectorElement* FindDE(const IDetectorElement* de, NameHistory_t& name_history);
   int InHistory(const IDetectorElement* de, const NameHistory_t& name_history);

   const DetectorElement* SensDetElem(const G4TouchableHistory& hist);
   int  SensDetId(const DetectorElement& de);

 
private:
  
  ///no default constructor
  GiGaRunActionExport();
  /// no copy constructor 
  GiGaRunActionExport( const GiGaRunActionExport& );  
  /// no assignement 
  GiGaRunActionExport& operator=( const GiGaRunActionExport& );


private:

  /// PackedIdParameterName : name of user paramater of the counted
  /// detector element which holds the packed, globally unique PMT
  /// ID.
  std::string m_idParameter;

  //  identifier values gleaned from standard full volume traverse
  //  as used by COLLADA exporter 
  std::vector<int> m_pvid; 
  std::vector<std::string> m_pvname;   // for debug, not identity matching 
  std::vector<G4AffineTransform> m_transform ; 

  std::string m_schemaPath ;
  std::string m_outFilePath ;     

 
};
#endif ///< GIGA_GIGARUNACTIONEXPORT_H
