#ifndef GIGA_GIGARUNACTIONGDML_H 
#define GIGA_GIGARUNACTIONGDML_H 1

#include "GiGa/GiGaRunActionBase.h"

/** @class GiGaRunActionGDML GiGaRunActionGDML.h
 *  
 *  A concrete Run Action. 
 *  It exports Geant4 geometry into a GDML file at the BeginOfRun
 *
 *  Updated from http://svn.cern.ch/guest/lhcb/packages/trunk/Sim/GDMLG4Writer/src/GDMLRunAction.h 
 *
 *  @author Simon Blytyh
 *  @date   
 */

class GiGaRunActionGDML: public virtual GiGaRunActionBase
{
  /// friend factory for instantiation
  //friend class GiGaFactory<GiGaRunActionGDML>;
  
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
  GiGaRunActionGDML
  ( const std::string& type   ,
    const std::string& name   ,
    const IInterface*  parent ) ;
  
  // destructor (virtual and protected)
  virtual ~GiGaRunActionGDML( );
  
private:
  
  ///no default constructor
  GiGaRunActionGDML();
  /// no copy constructor 
  GiGaRunActionGDML( const GiGaRunActionGDML& );  
  /// no assignement 
  GiGaRunActionGDML& operator=( const GiGaRunActionGDML& );

private:

  std::string m_schemaPath ;
  std::string m_outFilePath ;     

 
};
#endif ///< GIGA_GIGARUNACTIONGDML_H
