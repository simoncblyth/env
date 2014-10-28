#ifndef DSCHROMARUNACTION_H 
#define DSCHROMARUNACTION_H 1

#include "GaudiAlg/GaudiTool.h"
#include "GiGa/GiGaRunActionBase.h"

/** @class DsChromaRunAction 
 *  
 *  A concrete Run Action. 
 *
 *  @author Simon Blyth
 *  @date   
 */

class DsChromaRunAction : public virtual GiGaRunActionBase
{
public:

  void BeginOfRunAction ( const G4Run* run );
  
  void EndOfRunAction   ( const G4Run* run );
  
  DsChromaRunAction  
  ( const std::string& type   ,
    const std::string& name   ,
    const IInterface*  parent ) ;
  
  virtual ~DsChromaRunAction( );

private:
  
  DsChromaRunAction();
  DsChromaRunAction( const DsChromaRunAction& );  
  DsChromaRunAction& operator=( const DsChromaRunAction& );

private:

  // configuration of G4DAEChroma singleton
  std::string m_transport ;
  std::string m_sensdet ;
  std::string m_geometry ;

  /// PackedIdParameterName : name of user paramater of the counted
  /// detector element which holds the packed, globally unique PMT
  /// ID.
  std::string m_idParameter;

  /// TouchableToDetelem : the ITouchableToDetectorElement to use to
  /// resolve sensor ID.
  std::string m_t2deName;

 
};
#endif
