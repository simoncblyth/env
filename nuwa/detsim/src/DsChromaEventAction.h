#ifndef DSCHROMAEVENTACTION_H 
#define DSCHROMAEVENTACTION_H 1

#include "GaudiAlg/GaudiTool.h"
#include "GiGa/GiGaEventActionBase.h"

/** @class DsChromaEventAction 
 *  
 *  A concrete Event Action. 
 *
 *  @author Simon Blyth
 *  @date   
 */

class DsChromaEventAction : public virtual GiGaEventActionBase
{
public:

  void BeginOfEventAction ( const G4Event* event );
  
  void EndOfEventAction   ( const G4Event* event );
  
  DsChromaEventAction  
  ( const std::string& type   ,
    const std::string& name   ,
    const IInterface*  parent ) ;
  
  virtual ~DsChromaEventAction( );

private:
  
  DsChromaEventAction();
  DsChromaEventAction( const DsChromaEventAction& );  
  DsChromaEventAction& operator=( const DsChromaEventAction& );

private:
 
};
#endif
